import os
import re
import math
import streamlit as st
import pandas as pd
import folium
from folium.plugins import Fullscreen  # Para pantalla completa
from streamlit_folium import folium_static
import plotly.express as px
import datetime  # Para trabajar con objetos time
import google.generativeai as genai
import numpy as np

# ----------------------------
# CONFIGURACIÓN DE GEMINI AI
# ----------------------------
st.sidebar.header("Configururación de Gemini AI")
gemini_api_key = st.sidebar.text_input("Ingrese su API key de Gemini", type="password")
gemini_enabled = False
if gemini_api_key:
    genai.configure(api_key=gemini_api_key)
    generation_config = {
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 65536,
        "response_mime_type": "text/plain",
    }
    gemini_model = genai.GenerativeModel(
        model_name="gemini-2.0-flash",
        generation_config=generation_config,
    )
    gemini_enabled = True
else:
    st.sidebar.warning("Ingrese su API key de Gemini para habilitar el análisis automático.")

# ------------------------------------------------------
# CARGA DE SENSOR MASTER LIST DESDE CSV O EXCEL
# ------------------------------------------------------
st.sidebar.header("Sensor Master List")
master_file = st.sidebar.file_uploader("Cargar Sensor Master List (CSV o Excel)", type=["csv", "xls", "xlsx"], key="master_file")

sensor_master_df = None
if master_file is not None:
    file_name = master_file.name.lower()
    try:
        if file_name.endswith('.csv'):
            sensor_master_df = pd.read_csv(master_file)
            sensor_master_df['Sheet'] = "CSV"  # Indicar que proviene de CSV
        else:
            sheets_dict = pd.read_excel(master_file, sheet_name=None)
            sensor_master_df = pd.concat(
                [df.assign(Sheet=sheet) for sheet, df in sheets_dict.items()],
                ignore_index=True
            )
    except Exception as e:
        st.sidebar.error(f"Error al leer el archivo: {e}")

required_cols = ["Property ID in AVL packet", "Property Name", "Units", "Description"]
if sensor_master_df is not None:
    missing = [col for col in required_cols if col not in sensor_master_df.columns]
    if missing:
        st.sidebar.error(f"El archivo de Sensor Master List no contiene las siguientes columnas: {missing}")
    else:
        st.sidebar.success("Sensor Master List cargada correctamente.")

sensor_master_mapping = {}
if sensor_master_df is not None:
    for idx, row in sensor_master_df.iterrows():
        key = str(row["Property ID in AVL packet"]).strip()
        name = row["Property Name"]
        units = row["Units"]
        if pd.notna(units) and str(units).strip() != "":
            full_name = f"{name} (io_{key}) [{units}]"
        else:
            full_name = f"{name} (io_{key})"
        sensor_master_mapping[key] = {
            "name": full_name,
            "description": row["Description"]
        }

with st.sidebar.expander("Ver Sensor Master List"):
    if sensor_master_df is not None:
        st.dataframe(sensor_master_df)
    else:
        st.write("No se cargó un archivo de Sensor Master List. Se usará la versión por defecto.")

# ------------------------------------------------------
# CONFIGURACIÓN DEL PERÍODO DE ANÁLISIS RETROACTIVO
# ------------------------------------------------------
st.sidebar.header("Análisis Retroactivo")
retro_period = st.sidebar.number_input(
    "Periodo de análisis retroactivo (minutos)", 
    value=15, 
    min_value=1, 
    step=1,
    help="Tiempo hacia atrás, desde el último reporte, para analizar el comportamiento de los sensores."
)

# ------------------------------------------------------
# INFORMACIÓN ADICIONAL DE GPS
# ------------------------------------------------------
st.sidebar.header("Información Adicional de GPS")
gps_knowledge = st.sidebar.text_area(
    "Proporciona aquí información extra sobre el GPS, características, posibles fallas, etc.",
    help="Este texto se usará como conocimiento adicional para la IA."
)

# ------------------------------------------------------
# FUNCIONES AUXILIARES
# ------------------------------------------------------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000  # Radio de la Tierra en metros
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def convert_timestamp(ts):
    try:
        ts_val = float(ts)
        if ts_val > 1e11:
            dt = pd.to_datetime(ts_val, unit='ms', utc=True)
        else:
            dt = pd.to_datetime(ts_val, unit='s', utc=True)
        local_tz = "America/Mexico_City"
        dt_local = dt.tz_convert(local_tz).tz_localize(None)
        return dt_local
    except Exception as e:
        st.error(f"Error convirtiendo timestamp: {ts} ({e})")
        return pd.NaT

def rename_sensor_columns(df, master_mapping):
    new_columns = {}
    for col in df.columns:
        if col.startswith("io_"):
            numeric_part = col[3:].strip()
            if numeric_part in master_mapping:
                new_columns[col] = master_mapping[numeric_part]["name"]
    if new_columns:
        df.rename(columns=new_columns, inplace=True)

def parse_reg_data(uploaded_file):
    data = []
    if uploaded_file is not None:
        for line in uploaded_file:
            line = line.decode("utf-8").strip()
            if line.startswith("REG;"):
                parts = line.split(";")
                try:
                    if len(parts) < 4:
                        continue
                    timestamp = parts[1].strip()
                    lon = float(parts[2].strip())
                    lat = float(parts[3].strip())
                    record = {"timestamp": timestamp, "longitude": lon, "latitude": lat}
                    for field in parts[4:]:
                        if ":" in field:
                            pairs = field.split(",")
                            for pair in pairs:
                                if ":" in pair:
                                    key, value = pair.split(":", 1)
                                    key = key.strip().lower()
                                    value = value.strip().strip('"')
                                    try:
                                        record[key] = float(value)
                                    except ValueError:
                                        record[key] = value
                    data.append(record)
                except Exception as e:
                    print(f"Error procesando la línea: {line} - {e}")
                    continue
    return pd.DataFrame(data)

def get_pre_termination_data(df, retro_minutes):
    df = df.sort_values("datetime").reset_index(drop=True)
    last_time = df['datetime'].max()
    start_time = last_time - pd.Timedelta(minutes=retro_minutes)
    pre_term_df = df[df['datetime'] >= start_time]
    return pre_term_df, start_time, last_time

def create_sensor_map(data_points, selected_parameter, opacity,
                      color_min=None, color_max=None, custom_intervals=None,
                      pre_term_indices=None, sensor_master_mapping=None):
    if data_points.empty:
        st.warning("No hay datos para mostrar en el mapa.")
        return None
    if selected_parameter is None:
        st.error("No se ha seleccionado un parámetro válido.")
        return None

    avg_lat = data_points['latitude'].mean()
    avg_lon = data_points['longitude'].mean()
    sensor_map = folium.Map(location=[avg_lat, avg_lon], zoom_start=14)
    Fullscreen(position='topright').add_to(sensor_map)

    basemap_options = {
        'OpenStreetMap': folium.TileLayer('openstreetmap'),
        'Stamen Toner': folium.TileLayer('stamentoner', 
                          attr='Map tiles by Stamen Design, CC BY 3.0 — Map data © OpenStreetMap'),
        'Stamen Terrain': folium.TileLayer('stamenterrain', 
                          attr='Map tiles by Stamen Design, CC BY 3.0 — Map data © OpenStreetMap'),
        'CartoDB Positron': folium.TileLayer('cartodbpositron'),
        'CartoDB Dark Matter': folium.TileLayer('cartodbdarkmatter'),
    }
    selected_basemap = st.selectbox("Selecciona mapa base:", list(basemap_options.keys()), key="map_base")
    basemap_options[selected_basemap].add_to(sensor_map)

    def get_io_number(col_name):
        pattern = r"io_(\d+)"
        match = re.search(pattern, col_name)
        if match:
            return match.group(1)
        return None

    is_numeric = pd.api.types.is_numeric_dtype(data_points[selected_parameter])
    if custom_intervals and len(custom_intervals) > 0 and is_numeric:
        def get_color(value):
            try:
                val = float(value)
                for interval in custom_intervals:
                    if interval["min"] <= val <= interval["max"]:
                        return interval["color"]
                return "gray"
            except:
                return "gray"

        for idx, row in data_points.iterrows():
            try:
                sensor_value = float(row[selected_parameter])
            except:
                sensor_value = row[selected_parameter]
            color = get_color(sensor_value)
            popup_html = f"<b>{selected_parameter}:</b> {sensor_value}<br>"
            io_num = get_io_number(selected_parameter)
            if io_num and sensor_master_mapping and io_num in sensor_master_mapping:
                desc = sensor_master_mapping[io_num].get("description", "")
                popup_html += f"<i>{desc}</i><br>"
            if pre_term_indices is not None and idx in pre_term_indices:
                folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    radius=6, color="orange", fill=True, fill_color="orange",
                    fill_opacity=opacity, popup=folium.Popup(popup_html, max_width=300)
                ).add_to(sensor_map)
            else:
                folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    radius=5, color=color, fill=True, fill_color=color,
                    fill_opacity=opacity, popup=folium.Popup(popup_html, max_width=300)
                ).add_to(sensor_map)
        legend_items = ""
        for interval in custom_intervals:
            legend_items += f'''
            <div style="display:flex; align-items:center; margin-bottom:2px;">
                <div style="background-color:{interval["color"]}; width:20px; height:10px; margin-right:5px;"></div>
                {interval["min"]} - {interval["max"]}
            </div>
            '''
        legend_html = f'''
        <div style="position: fixed; top: 50px; right: 50px; width: 180px; border:2px solid grey; z-index:9999; font-size:14px; background-color: white; padding: 10px;">
            <b>Intervalos</b><br>{legend_items}
        </div>
        '''
        sensor_map.get_root().html.add_child(folium.Element(legend_html))
    else:
        if not is_numeric:
            for idx, row in data_points.iterrows():
                sensor_value = row[selected_parameter]
                popup_html = f"<b>{selected_parameter}:</b> {sensor_value}<br>"
                io_num = get_io_number(selected_parameter)
                if io_num and sensor_master_mapping and io_num in sensor_master_mapping:
                    desc = sensor_master_mapping[io_num].get("description", "")
                    popup_html += f"<i>{desc}</i><br>"
                color = "blue" if pre_term_indices is None or idx not in pre_term_indices else "orange"
                folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    radius=5, color=color, fill=True, fill_color=color,
                    fill_opacity=opacity, popup=folium.Popup(popup_html, max_width=300)
                ).add_to(sensor_map)
        else:
            try:
                default_min = data_points[selected_parameter].quantile(0.01)
                default_max = data_points[selected_parameter].quantile(0.99)
            except Exception as e:
                st.error(f"Error al calcular rangos para {selected_parameter}: {e}")
                return sensor_map

            min_val = color_min if color_min is not None and color_max is not None and color_min < color_max else default_min
            max_val = color_max if color_min is not None and color_max is not None and color_min < color_max else default_max

            colorscale = px.colors.sequential.Viridis
            for idx, row in data_points.iterrows():
                try:
                    sensor_value = float(row[selected_parameter])
                    if max_val == min_val:
                        color = colorscale[0]
                    else:
                        normalized_value = (sensor_value - min_val) / (max_val - min_val)
                        normalized_value = max(0, min(1, normalized_value))
                        index = int(normalized_value * (len(colorscale) - 1))
                        color = colorscale[index]
                except (ValueError, TypeError):
                    color = 'gray'
                    sensor_value = row[selected_parameter]
                popup_html = f"<b>{selected_parameter}:</b> {sensor_value}<br>"
                io_num = get_io_number(selected_parameter)
                if io_num and sensor_master_mapping and io_num in sensor_master_mapping:
                    desc = sensor_master_mapping[io_num].get("description", "")
                    popup_html += f"<i>{desc}</i><br>"
                if pre_term_indices is not None and idx in pre_term_indices:
                    folium.CircleMarker(
                        location=[row['latitude'], row['longitude']],
                        radius=6, color="orange", fill=True, fill_color="orange",
                        fill_opacity=opacity, popup=folium.Popup(popup_html, max_width=300)
                    ).add_to(sensor_map)
                else:
                    folium.CircleMarker(
                        location=[row['latitude'], row['longitude']],
                        radius=5, color=color, fill=True, fill_color=color,
                        fill_opacity=opacity, popup=folium.Popup(popup_html, max_width=300)
                    ).add_to(sensor_map)
            stops = 5
            stops_html = ""
            for i in range(stops):
                val_stop = min_val + i * (max_val - min_val) / (stops - 1)
                norm = i / (stops - 1)
                idx_color = int(norm * (len(colorscale) - 1))
                color_stop = colorscale[idx_color]
                stops_html += f'''
                <div style="display:flex; align-items:center;">
                  <div style="background-color:{color_stop}; width:20px; height:10px; margin-right:5px;"></div>
                  {val_stop:.2f}
                </div>
                '''
            legend_html = f'''
            <div style="position: fixed; top: 50px; right: 50px; width: 150px; border:2px solid grey; z-index:9999; font-size:14px; background-color: white; padding: 10px;">
                <b>Escala de color</b><br>{stops_html}
            </div>
            '''
            sensor_map.get_root().html.add_child(folium.Element(legend_html))

    last_row = data_points.iloc[-1]
    folium.Marker(
        location=[last_row['latitude'], last_row['longitude']],
        icon=folium.Icon(color="red", icon="flag"),
        popup=folium.Popup("Último reporte", max_width=200)
    ).add_to(sensor_map)

    return sensor_map

def create_basic_map(data_points, opacity, pre_term_indices=None):
    if data_points.empty:
        st.warning("No hay datos para mostrar en el mapa.")
        return None
    avg_lat = data_points['latitude'].mean()
    avg_lon = data_points['longitude'].mean()
    sensor_map = folium.Map(location=[avg_lat, avg_lon], zoom_start=14)
    Fullscreen(position='topright').add_to(sensor_map)
    basemap_options = {
        'OpenStreetMap': folium.TileLayer('openstreetmap'),
        'Stamen Toner': folium.TileLayer('stamentoner', 
                             attr='Map tiles by Stamen Design, CC BY 3.0 — Map data © OpenStreetMap'),
        'Stamen Terrain': folium.TileLayer('stamenterrain', 
                             attr='Map tiles by Stamen Design, CC BY 3.0 — Map data © OpenStreetMap'),
        'CartoDB Positron': folium.TileLayer('cartodbpositron'),
        'CartoDB Dark Matter': folium.TileLayer('cartodbdarkmatter'),
    }
    selected_basemap = st.selectbox("Selecciona mapa base:", list(basemap_options.keys()), key="basic_map")
    basemap_options[selected_basemap].add_to(sensor_map)

    for idx, row in data_points.iterrows():
        popup_html = f"<b>Latitud:</b> {row['latitude']}<br><b>Longitud:</b> {row['longitude']}"
        if pre_term_indices is not None and idx in pre_term_indices:
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=6, color="orange", fill=True, fill_color="orange",
                fill_opacity=opacity, popup=folium.Popup(popup_html, max_width=300)
            ).add_to(sensor_map)
        else:
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=5, color='blue', fill=True, fill_color='blue',
                fill_opacity=opacity, popup=folium.Popup(popup_html, max_width=300)
            ).add_to(sensor_map)

    last_row = data_points.iloc[-1]
    folium.Marker(
        location=[last_row['latitude'], last_row['longitude']],
        icon=folium.Icon(color="red", icon="flag"),
        popup=folium.Popup("Último reporte", max_width=200)
    ).add_to(sensor_map)
    return sensor_map

def convert_markdown_to_pdf(markdown_text):
    """
    Función modificada para convertir texto a PDF sin utilizar el módulo 'markdown'.
    Se envuelve el texto en una etiqueta <pre> para preservar el formato.
    """
    from xhtml2pdf import pisa
    from io import BytesIO

    # Se asume que el texto recibido es markdown; en lugar de parsearlo, se envuelve en <pre>
    html_content = f"<pre>{markdown_text}</pre>"
    html = f"""
    <html>
    <head>
      <meta charset="UTF-8">
      <style>
        body {{
            font-family: Arial, sans-serif;
            font-size: 12pt;
            line-height: 1.5;
            margin: 20px;
        }}
        pre {{
            white-space: pre-wrap;
        }}
      </style>
    </head>
    <body>
    {html_content}
    </body>
    </html>
    """
    pdf = BytesIO()
    pisa_status = pisa.CreatePDF(html, dest=pdf)
    if pisa_status.err:
        st.error("Error al generar el PDF")
        return None
    return pdf.getvalue()

def compute_slope(series, timestamps):
    if len(series) < 2:
        return np.nan
    t0 = timestamps.min().timestamp()
    times = np.array([t.timestamp() - t0 for t in timestamps])
    return np.polyfit(times, series, 1)[0]

# ------------------------------------------------------
# AGREGAR CÁLCULO DE LA TASA DE ANOMALÍAS
# ------------------------------------------------------
def calcular_tasa_anomalias(pre_term_df, sensor):
    count = pre_term_df[sensor].count()
    q1 = pre_term_df[sensor].quantile(0.25)
    q3 = pre_term_df[sensor].quantile(0.75)
    iqr = q3 - q1
    outliers = ((pre_term_df[sensor] < (q1 - 1.5 * iqr)) | (pre_term_df[sensor] > (q3 + 1.5 * iqr))).sum()
    if count > 0:
        tasa = (outliers / count) * 100
    else:
        tasa = 0
    return round(tasa, 2)

# ------------------------------------------------------
# ORGANIZACIÓN EN PESTAÑAS: "Análisis" y "Conversación"
# ------------------------------------------------------
tabs = st.tabs(["Análisis", "Conversación"])

# ------------------- Pestaña de Análisis -------------------
with tabs[0]:
    st.title("Sensor Insights - Análisis del Corte de Reporte")
    uploaded_file = st.file_uploader("Carga un archivo de datos (TXT o WLN) de reportes", type=["txt", "wln"], key="data_file")
    if uploaded_file is not None:
        df = parse_reg_data(uploaded_file)
        if df.empty:
            st.error("No se pudieron procesar los datos del archivo. Verifica el formato.")
            st.stop()

        if 'timestamp' in df.columns:
            df['datetime'] = df['timestamp'].apply(convert_timestamp)
            st.write("Rango de fechas en los datos:", df['datetime'].min().date(), "a", df['datetime'].max().date())
            last_message = df['datetime'].max()
            st.markdown(f"### Último mensaje detectado: **{last_message}**")
            
            date_range = st.date_input("Seleccione rango de fechas para analizar",
                                       value=[df['datetime'].min().date(), df['datetime'].max().date()])
            if isinstance(date_range, list) and len(date_range) == 2:
                start_date, end_date = date_range
                df = df[(df['datetime'].dt.date >= start_date) & (df['datetime'].dt.date <= end_date)]
                if df.empty:
                    st.error("No hay datos en el rango de fechas seleccionado.")
            
            st.write("Filtrar por hora:")
            start_time = st.time_input("Hora de inicio", value=datetime.time(0, 0))
            end_time = st.time_input("Hora de fin", value=datetime.time(23, 59))
            df = df[(df['datetime'].dt.time >= start_time) & (df['datetime'].dt.time <= end_time)]
            if df.empty:
                st.error("No hay datos en el rango de horas seleccionado.")

        if sensor_master_mapping:
            rename_sensor_columns(df, sensor_master_mapping)

        df = df.sort_values("datetime").reset_index(drop=True)
        pre_term_df, retro_start, last_time = get_pre_termination_data(df, retro_period)
        pre_term_indices = pre_term_df.index.tolist()

        st.subheader("Análisis Retroactivo")
        st.write(f"Se analizarán los datos desde {retro_start} hasta el último reporte ({last_time}).")
        st.dataframe(pre_term_df.head())

        available_columns = [col for col in df.columns if col not in ['latitude', 'longitude', 'timestamp', 'datetime']]
        sensor_stats_pre = []
        for col in available_columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                overall_mean = df[col].mean()
                overall_median = df[col].median()
                overall_std = df[col].std()
                overall_min = df[col].min()
                overall_max = df[col].max()
                overall_p5 = df[col].quantile(0.05)
                overall_p95 = df[col].quantile(0.95)
                pre_mean = pre_term_df[col].mean()
                pre_median = pre_term_df[col].median()
                pre_std = pre_term_df[col].std()
                pre_min = pre_term_df[col].min()
                pre_max = pre_term_df[col].max()
                pre_p5 = pre_term_df[col].quantile(0.05)
                pre_p95 = pre_term_df[col].quantile(0.95)
                diff_mean = overall_mean - pre_mean
                diff_median = overall_median - pre_median
                pct_diff_mean = (diff_mean / overall_mean * 100) if overall_mean != 0 else 0
                pct_diff_median = (diff_median / overall_median * 100) if overall_median != 0 else 0
                overall_slope = compute_slope(df[col].dropna(), df['datetime'][df[col].notna()])
                pre_slope = compute_slope(pre_term_df[col].dropna(), pre_term_df['datetime'][pre_term_df[col].notna()])
                diff_slope = overall_slope - pre_slope if (not np.isnan(overall_slope) and not np.isnan(pre_slope)) else np.nan
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                total_outliers = ((df[col] < (q1 - 1.5 * iqr)) | (df[col] > (q3 + 1.5 * iqr))).sum()
                pre_outliers = ((pre_term_df[col] < (q1 - 1.5 * iqr)) | (pre_term_df[col] > (q3 + 1.5 * iqr))).sum()
                tasa_anomalias = calcular_tasa_anomalias(pre_term_df, col)
                sensor_stats_pre.append({
                    "Sensor": col,
                    "Promedio Total": overall_mean,
                    "Promedio Retroactivo": pre_mean,
                    "Diferencia Promedio": diff_mean,
                    "Diferencia Promedio (%)": pct_diff_mean,
                    "Mediana Total": overall_median,
                    "Mediana Retroactivo": pre_median,
                    "Diferencia Mediana": diff_median,
                    "Diferencia Mediana (%)": pct_diff_median,
                    "Mínimo Total": overall_min,
                    "Mínimo Retroactivo": pre_min,
                    "Máximo Total": overall_max,
                    "Máximo Retroactivo": pre_max,
                    "Desviación Estándar Total": overall_std,
                    "Desviación Estándar Retroactivo": pre_std,
                    "Percentil 5 Total": overall_p5,
                    "Percentil 95 Total": overall_p95,
                    "Percentil 5 Retroactivo": pre_p5,
                    "Percentil 95 Retroactivo": pre_p95,
                    "Pendiente Total": overall_slope,
                    "Pendiente Retroactivo": pre_slope,
                    "Diferencia de Pendiente": diff_slope,
                    "Outliers Totales": total_outliers,
                    "Outliers Retroactivos": pre_outliers,
                    "Tasa Anomalías Retro (%)": tasa_anomalias
                })
        stats_pre_df = pd.DataFrame(sensor_stats_pre)
        st.subheader("Estadísticas Mejoradas de Sensores en el Período Retroactivo")
        st.dataframe(stats_pre_df)
        st.write("Datos procesados (primeros registros):", df.head())
        opacity = st.slider("Opacidad de los marcadores", 0.1, 1.0, 0.7)

        custom_intervals = None
        color_min = color_max = None
        selected_parameter = None
        if available_columns:
            selected_parameter = st.selectbox("Selecciona el sensor para visualizar:", available_columns)
            if pd.api.types.is_numeric_dtype(df[selected_parameter]):
                sensor_min = df[selected_parameter].min()
                sensor_max = df[selected_parameter].max()
                st.write("Rango de valores en los datos para", selected_parameter, ":", sensor_min, "a", sensor_max)
                use_custom = st.checkbox("Usar intervalos de color personalizados")
                if not use_custom:
                    color_min = st.number_input("Valor mínimo para escala de color", value=float(df[selected_parameter].quantile(0.01)))
                    color_max = st.number_input("Valor máximo para escala de color", value=float(df[selected_parameter].quantile(0.99)))
                else:
                    num_intervals = st.number_input("Número de intervalos personalizados", min_value=1, max_value=10, step=1, value=2)
                    custom_intervals = []
                    default_step = (sensor_max - sensor_min) / num_intervals if num_intervals > 0 else (sensor_max - sensor_min)
                    st.markdown("### Defina los intervalos y seleccione el color para cada uno")
                    for i in range(int(num_intervals)):
                        default_min = sensor_min + i * default_step
                        default_max = sensor_min + (i + 1) * default_step
                        col_enabled, col1, col2, col3 = st.columns([1, 3, 3, 3])
                        enabled = col_enabled.checkbox(f"Activar Intervalo {i+1}", key=f"enabled_{i}", value=True)
                        with col1:
                            min_val = st.number_input(f"Valor mínimo (Intervalo {i+1})", key=f"min_{i}", value=float(default_min))
                        with col2:
                            max_val = st.number_input(f"Valor máximo (Intervalo {i+1})", key=f"max_{i}", value=float(default_max))
                        with col3:
                            color_val = st.color_picker(f"Color (Intervalo {i+1})", key=f"color_{i}", value="#0000ff")
                        if enabled:
                            custom_intervals.append({"min": min_val, "max": max_val, "color": color_val})
            else:
                selected_parameter = selected_parameter
        else:
            st.info("No hay columnas de sensor disponibles en el archivo de reportes. Se mostrará el mapa con las ubicaciones únicamente.")

        if not available_columns:
            sensor_map = create_basic_map(df, opacity, pre_term_indices=pre_term_indices)
        else:
            sensor_map = create_sensor_map(
                data_points=df,
                selected_parameter=selected_parameter,
                opacity=opacity,
                color_min=color_min,
                color_max=color_max,
                custom_intervals=custom_intervals,
                pre_term_indices=pre_term_indices,
                sensor_master_mapping=sensor_master_mapping
            )
        if sensor_map:
            folium_static(sensor_map)

        if available_columns and selected_parameter:
            st.subheader(f"Histograma de {selected_parameter}")
            try:
                fig = px.histogram(df, x=selected_parameter, nbins=50, marginal="box", color_discrete_sequence=['#636EFA'])
                fig.update_layout(xaxis_title=selected_parameter, yaxis_title="Frecuencia", bargap=0.1)
                st.plotly_chart(fig)
            except (ValueError, KeyError) as e:
                st.error(f"Error al crear el histograma: {e}. Asegúrate de que '{selected_parameter}' sea una columna numérica.")

        if st.checkbox("Mostrar datos en tabla"):
            st.subheader("Datos")
            st.dataframe(df)

        if gemini_enabled:
            include_hypotheses = st.sidebar.checkbox("Incluir hipótesis de falla en el análisis", value=False)
            if st.button("Obtener Análisis Automático con Gemini AI"):
                if include_hypotheses:
                    hypothesis_section = (
                        "3.  **Hipótesis de Falla (Análisis Causal):**\n"
                        "    - Presenta una lista de hipótesis sobre las posibles causas de la interrupción. Para cada hipótesis, menciona brevemente la función de los sensores o componentes técnicos involucrados y explica cómo influyen en el contexto actual analizado.\n\n"
                    )
                    prompt = f"""
Eres un analista de datos experto en sistemas telemáticos y dispositivos GPS. Analiza la información proporcionada para identificar las causas de una interrupción en la transmisión de datos GPS. El análisis debe ser exhaustivo, crítico y proactivo. Sigue esta estructura utilizando listas:

1. **Resumen Ejecutivo (Crítico):**
   - Resume de forma concisa los hallazgos más relevantes.
2. **Análisis de Sensores Afectados:**
   - Para cada sensor, indica:
       • Sensor Técnico y Sensor Común.
       • Breve descripción de su función y cómo influye en el contexto actual.
       • Valores promedio global y en el período retroactivo.
       • Diferencia de promedios y diferencia mínima.
       • Tasa de anomalías durante el período (porcentaje de valores atípicos).
3. {hypothesis_section}
4. **Recomendaciones (Plan de Acción):**
   - Proporciona una lista de acciones concretas para verificar las hipótesis planteadas.
5. **Análisis de Tendencias (Opcional y Avanzado):**
   - Indica si existen patrones recurrentes en el tiempo relacionados con las anomalías.

Utiliza un lenguaje técnico preciso y accesible.
"""
                else:
                    hypothesis_section = (
                        "3. **Recomendaciones de Verificación y Acción:**\n"
                        "   - Presenta una lista de recomendaciones detalladas para identificar la causa de la interrupción. Para cada recomendación, menciona brevemente la función de los sensores o componentes técnicos relevantes y cómo influyen en el análisis.\n\n"
                    )
                    prompt = f"""
Eres un analista de datos experto en sistemas telemáticos y dispositivos GPS. Analiza la información proporcionada para identificar posibles causas de una interrupción en la transmisión de datos GPS. El análisis debe ser exhaustivo, crítico y orientado a recomendaciones prácticas. Sigue esta estructura utilizando listas:

1. **Resumen Ejecutivo (Crítico):**
   - Resume de forma concisa los hallazgos más relevantes.
2. **Análisis de Sensores Afectados:**
   - Para cada sensor, indica:
       • Sensor Técnico y Sensor Común.
       • Breve descripción de su función y de cómo influye en el contexto actual.
       • Valores promedio global y en el período retroactivo.
       • Diferencia de promedios y diferencia mínima.
       • Tasa de anomalías durante el período (porcentaje de valores atípicos).
3. {hypothesis_section}
4. **Análisis de Tendencias (Opcional y Avanzado):**
   - Indica si existen patrones recurrentes en el tiempo relacionados con las anomalías.

Utiliza un lenguaje técnico preciso y accesible.
"""
                if sensor_master_df is not None:
                    sensor_info = "Listado de sensores:\n• Para cada sensor, menciona Sensor Técnico, Sensor Común, Unidades y una breve descripción de su función y relevancia en el contexto analizado.\n"
                    for idx, row in sensor_master_df.iterrows():
                        property_id = str(row["Property ID in AVL packet"]).strip()
                        sensor_tecnico = f"io_{property_id}"
                        sensor_comun = row["Property Name"]
                        unidades = row["Units"] if pd.notna(row["Units"]) and str(row["Units"]).strip() != "" else "N/A"
                        descripcion = row["Description"]
                        sensor_info += f"- {sensor_tecnico} | {sensor_comun} | {unidades} | {descripcion}\n"
                else:
                    sensor_info = "No se proporcionó información adicional de sensores."
                prompt += f"\n\n**Información Adicional de Sensores:**\n{sensor_info}\n"
                if 'chat_session' not in st.session_state:
                    st.session_state['chat_session'] = gemini_model.start_chat(history=[])
                chat_session = st.session_state['chat_session']
                response = chat_session.send_message(prompt)
                analysis_text = response.text
                st.subheader("Análisis Automático de Gemini AI")
                st.markdown(analysis_text, unsafe_allow_html=True)
                pdf_bytes = convert_markdown_to_pdf(analysis_text)
                if pdf_bytes:
                    st.download_button("Descargar análisis a PDF", data=pdf_bytes, file_name="analisis_gemini.pdf", mime="application/pdf")
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button("Descargar datos como CSV", csv_data, "datos_sensores.csv", "text/csv", key='download-csv')

# ------------------- Pestaña de Conversación -------------------
with tabs[1]:
    st.subheader("Conversación Iterativa con Gemini AI")
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []
    user_input = st.text_input("Escribe tu pregunta o comentario:", key="user_input")
    if st.button("Enviar mensaje", key="send_button"):
        if user_input:
            if 'chat_session' not in st.session_state:
                st.session_state['chat_session'] = gemini_model.start_chat(history=[])
            chat_session = st.session_state['chat_session']
            response = chat_session.send_message(user_input)
            st.session_state['chat_history'].append({"user": user_input, "ai": response.text})
    if st.session_state.get('chat_history'):
        st.markdown("### Historial de Conversación")
        for exchange in st.session_state['chat_history']:
            st.markdown(f"**Tú:** {exchange['user']}")
            st.markdown(f"**AI:** {exchange['ai']}")
