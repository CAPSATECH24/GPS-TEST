import os
import re
import math
import time  # Para las esperas en reintentos
import streamlit as st
import pandas as pd
import folium
from folium.plugins import Fullscreen  # Para pantalla completa
from streamlit_folium import folium_static
import plotly.express as px
import datetime  # Para trabajar con objetos time
import google.generativeai as genai
import numpy as np

# Para capturar las excepciones de la API de Google
import google.api_core.exceptions

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
# Función para reintentar el envío de mensajes
# ------------------------------------------------------
def send_message_with_retry(chat_session, message, max_retries=3, delay=5):
    """
    Envía un mensaje a la API de Gemini con reintentos en caso de error 429 (ResourceExhausted).
    - chat_session: objeto de sesión de chat con la API.
    - message: texto a enviar.
    - max_retries: número máximo de reintentos.
    - delay: tiempo de espera (segundos) entre reintentos.
    """
    for attempt in range(max_retries):
        try:
            response = chat_session.send_message(message)
            return response
        except google.api_core.exceptions.ResourceExhausted as e:
            # Error 429 o relacionado con límites de cuota
            st.warning(f"Recurso agotado (429). Reintentando en {delay} segundos... (intento {attempt+1}/{max_retries})")
            time.sleep(delay)
        except Exception as e:
            # Otros errores, se propaga la excepción
            raise e

    # Si se exceden los reintentos, lanzamos el error final
    raise google.api_core.exceptions.ResourceExhausted("Se ha agotado el número máximo de reintentos a la API de Gemini.")

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
gps_info = st.sidebar.text_area(
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
# ORGANIZACIÓN EN PESTAÑAS: "Análisis", "Conversación" y "Historial IA"
# ------------------------------------------------------
tabs = st.tabs(["Análisis", "Conversación", "Historial IA"])

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

        # ----------------------- GEMINI AI - Análisis Inicial -----------------------
        if gemini_enabled:
            include_hypotheses = st.sidebar.checkbox("Incluir hipótesis de falla en el análisis", value=False)
            if st.button("Obtener Análisis Automático con Gemini AI"):
                # Se definen las variables gps_info y sensor_info para integrar en el prompt
                sensor_info = ""
                if sensor_master_df is not None:
                    sensor_info = "Listado de sensores:\n"
                    for idx, row in sensor_master_df.iterrows():
                        property_id = str(row["Property ID in AVL packet"]).strip()
                        sensor_tecnico = f"io_{property_id}"
                        sensor_comun = row["Property Name"]
                        unidades = row["Units"] if pd.notna(row["Units"]) and str(row["Units"]).strip() != "" else "N/A"
                        descripcion = row["Description"]
                        sensor_info += f"- {sensor_tecnico} | {sensor_comun} | {unidades} | {descripcion}\n"
                else:
                    sensor_info = "No se proporcionó información adicional de sensores."

                # --- Prompt 1 Modificado (Análisis Inicial - Estilo Sherlock Holmes) ---
                prompt = f"""
Eres un analista de datos experto en sistemas telemáticos y dispositivos GPS, aplicando el método deductivo de Sherlock Holmes.

**Información Adicional y Contexto (Importante):**
{gps_info}

Tu objetivo es realizar una **investigación deductiva** para **comprender el comportamiento anómalo** detectado en los datos telemáticos y **determinar la causa más probable** de la interrupción en la transmisión de datos GPS. Aplica el método de Sherlock Holmes: **observación aguda, razonamiento lógico, inducción y generación de hipótesis fundamentadas.**

Sigue esta estructura utilizando listas:

1. **Resumen Ejecutivo (Crítico - Enfoque Deductivo):**
   - Resume de forma concisa los hallazgos más relevantes desde una perspectiva deductiva. **Identifica las posibles causas más probables de la interrupción, considerando tanto fallas internas como externas/físicas, basándote en la *evidencia inicial* disponible.**

2. **Análisis de Sensores Afectados (Observación Aguda y Detallada):**
   - Para cada sensor, realiza una **observación aguda y detallada** de su comportamiento en el período retroactivo. No pases por alto ningún detalle, incluso aquellos que parezcan insignificantes al principio. Indica:
       • Sensor Técnico y Sensor Común.
       • Breve descripción de su función y cómo influye en el contexto actual.
       • Valores promedio global y en el período retroactivo (como referencia inicial).
       • Diferencia de promedios y diferencia mínima (como referencia inicial).
       • Tasa de anomalías durante el período (porcentaje de valores atípicos - como indicador inicial).
       **Realiza un análisis profundo de la *evolución temporal* de los datos de cada sensor en el período retroactivo. Responde a las siguientes preguntas clave para una observación detallada:**
           * "¿Se observa alguna **tendencia específica** en los datos a lo largo del tiempo (aumento gradual, disminución repentina, fluctuaciones erráticas, patrones cíclicos)? Si es así, **describir la tendencia con precisión y el momento exacto en que se manifiesta.**"
           * "¿Existen **eventos específicos o cambios abruptos** en los valores de los sensores dentro del período retroactivo? **Identificar los momentos clave (fecha y hora en formato legible para humanos, zona horaria America/Mexico_City) y describir los cambios observados con detalle.**"
           * "¿Existen **correlaciones** entre diferentes sensores? ¿Cambios en un sensor se corresponden con cambios en otros? **Analizar posibles relaciones y dependencias entre los sensores, identificando patrones de comportamiento conjunto.**"
       **Al analizar los sensores y sus patrones temporales, piensa críticamente en cómo sus valores y *evolución en el tiempo* podrían indicar no solo un problema interno del GPS, sino también problemas de alimentación, conexión física, o factores externos.**

3. **Hipótesis de Falla (Análisis Causal Inductivo - Basado en la Evidencia Observada):**
   - Basándote en las **observaciones detalladas de los datos** (tendencias, eventos, correlaciones) realizadas en el punto anterior, genera una lista de posibles hipótesis de falla utilizando un enfoque **inductivo**.
     **Prioriza las hipótesis que mejor se ajusten a la evidencia observada y que expliquen de manera más coherente los patrones de comportamiento detectados en los sensores.**
     Considera un amplio espectro de posibilidades, incluyendo:
        * **Problemas de Conexión Física:** Cables, conectores, antena.
        * **Problemas de Alimentación Eléctrica:** Fuente, batería, cableado.
        * **Fallas Internas del GPS:** Módulo, procesador, firmware, software.
        * **Interferencia Externa:** Señal GPS (jamming), bloqueo por entorno.
        * **Manipulación o Desconexión Intencional:** Cables, desactivación.
       - **Para cada hipótesis, explica *cómo la evidencia observada* (patrones en los datos de los sensores, tendencias temporales, eventos específicos, correlaciones) *la apoya o la contradice*.** Sé específico en la conexión entre la evidencia y la hipótesis.
       - **Prioriza las hipótesis según su plausibilidad a la luz de la evidencia observada.**

4. **Recomendaciones (Plan de Acción Inicial - Orientado a la Investigación):**
   - Proporciona una lista de acciones concretas y **orientadas a la investigación** para **verificar las hipótesis planteadas** en el punto anterior. Las recomendaciones deben ser **lógicas y directamente derivadas de las hipótesis**.
     Asegúrate de que las recomendaciones incluyan pasos para investigar tanto fallas internas del GPS como problemas de conexión física y alimentación. Por ejemplo:
        * Inspección visual **detallada** de cables y conectores (buscar signos de daño, desgaste, conexiones sueltas).
        * Verificación de la alimentación eléctrica **en diferentes puntos del sistema** (fuente, GPS, cables).
        * Pruebas de continuidad en los cables **específicos** (alimentación, antena, datos).
        * Revisión **minuciosa** del estado de la antena GPS (posición, integridad física, conexión).
        * Diagnóstico interno del GPS **si es posible y relevante para las hipótesis.**
        * **Recopilación de información adicional relevante del contexto operativo** (ej: condiciones ambientales recientes, historial de la unidad, reportes de operadores).

5. **Análisis de Tendencias (Opcional y Avanzado - Patrones Temporales):**
   - Indica si existen **patrones recurrentes en el *tiempo*** (diarios, semanales, etc.) relacionados con las anomalías o comportamientos inusuales observados.
     **Considera si estos patrones temporales podrían estar relacionados con factores externos (ej: interferencia en ciertos momentos del día) o problemas de conexión intermitentes (ej: falsos contactos que varían con la vibración o temperatura).** Si se detectan patrones, descríbelos con precisión y propone posibles explicaciones basadas en estos patrones.

**Información Adicional de Sensores:**
{sensor_info}
                """
                if 'chat_session' not in st.session_state:
                    st.session_state['chat_session'] = gemini_model.start_chat(history=[])
                chat_session = st.session_state['chat_session']

                try:
                    response = send_message_with_retry(chat_session, prompt, max_retries=3, delay=5)
                    analysis_text = response.text
                    st.subheader("Análisis Automático de Gemini AI")
                    st.markdown(analysis_text, unsafe_allow_html=True)
                    # Almacenar el primer análisis en session_state para Historial IA
                    st.session_state['first_analysis'] = analysis_text
                    pdf_bytes = convert_markdown_to_pdf(analysis_text)
                    if pdf_bytes:
                        st.download_button("Descargar análisis a PDF", data=pdf_bytes, file_name="analisis_gemini.pdf", mime="application/pdf")
                except google.api_core.exceptions.ResourceExhausted as e:
                    st.error("Se ha excedido la cuota de la API de Gemini (error 429). Intente más tarde o revise su plan.")
                except Exception as e:
                    st.error(f"Ha ocurrido un error al enviar el mensaje: {e}")

        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button("Descargar datos como CSV", csv_data, "datos_sensores.csv", "text/csv", key='download-csv')

# ------------------- Pestaña de Conversación -------------------
with tabs[1]:
    st.subheader("Conversación Iterativa con Gemini AI")
    st.info("Para iniciar la conversación, asegúrate de haber ejecutado ambos análisis (Análisis Automático y Veredicto Final Integrado).")
    # Checkbox opcional para incluir el contexto histórico del archivo de reportes
    include_file_context = st.checkbox("Incluir contexto histórico de datos del archivo subido en la conversación", value=False)
    # Mostrar un mensaje si no se han ejecutado ambos análisis
    if 'first_analysis' not in st.session_state or 'second_analysis' not in st.session_state:
        st.warning("Aún no se han ejecutado ambos análisis. Ejecuta primero el Análisis Automático y luego el Veredicto Final Integrado para habilitar la conversación.")
    else:
        if 'chat_history' not in st.session_state:
            st.session_state['chat_history'] = []
        user_input = st.text_input("Escribe tu pregunta o comentario:", key="user_input")
        if st.button("Enviar mensaje", key="send_button"):
            if user_input:
                # Si se seleccionó incluir el contexto histórico y se dispone del archivo, se agrega esa información
                context_text = ""
                if include_file_context and 'data_file' in st.session_state and st.session_state.data_file is not None:
                    st.session_state.data_file.seek(0)
                    context_text = st.session_state.data_file.read().decode("utf-8")
                full_message = user_input + "\n\n" + context_text
                chat_session = st.session_state['chat_session']
                try:
                    response = send_message_with_retry(chat_session, full_message, max_retries=3, delay=5)
                    st.session_state['chat_history'].append({"user": user_input, "ai": response.text})
                except google.api_core.exceptions.ResourceExhausted as e:
                    st.error("Se ha excedido la cuota de la API de Gemini (error 429). Intente más tarde o revise su plan.")
                except Exception as e:
                    st.error(f"Ha ocurrido un error al enviar el mensaje: {e}")

        if st.session_state.get('chat_history'):
            st.markdown("### Historial de Conversación")
            for exchange in st.session_state['chat_history']:
                st.markdown(f"**Tú:** {exchange['user']}")
                st.markdown(f"**AI:** {exchange['ai']}")

# ------------------- Pestaña de Historial IA -------------------
with tabs[2]:
    st.subheader("Historial de Análisis de Gemini AI")
    if 'first_analysis' in st.session_state:
        st.markdown("#### Análisis Automático (Primer Análisis)")
        st.markdown(st.session_state['first_analysis'], unsafe_allow_html=True)
    else:
        st.info("El Análisis Automático aún no ha sido ejecutado.")
    st.markdown("---")
    if 'second_analysis' in st.session_state:
        st.markdown("#### Veredicto Final Integrado (Segundo Análisis)")
        st.markdown(st.session_state['second_analysis'], unsafe_allow_html=True)
    else:
        st.info("El Veredicto Final Integrado aún no ha sido ejecutado.")

# ------------------- Botón para Veredicto Final Integrado -------------------
if gemini_enabled:
    if st.button("Obtener Veredicto Final Integrado con Gemini AI", key="veredicto_button"):
        if 'first_analysis' not in st.session_state:
            st.error("Primero obtén el Análisis Automático en la pestaña de Análisis.")
        else:
            # Generar representación textual de los datos históricos
            if 'pre_term_df' not in locals():
                st.error("No hay datos en pre_term_df. Asegúrate de haber cargado el archivo y realizado el análisis previo.")
            else:
                retro_data_text = pre_term_df.to_csv(index=False)
                sensor_info = ""
                if sensor_master_df is not None:
                    sensor_info = "Listado de sensores:\n"
                    for idx, row in sensor_master_df.iterrows():
                        property_id = str(row["Property ID in AVL packet"]).strip()
                        sensor_tecnico = f"io_{property_id}"
                        sensor_comun = row["Property Name"]
                        unidades = row["Units"] if pd.notna(row["Units"]) and str(row["Units"]).strip() != "" else "N/A"
                        descripcion = row["Description"]
                        sensor_info += f"- {sensor_tecnico} | {sensor_comun} | {unidades} | {descripcion}\n"
                else:
                    sensor_info = "No se proporcionó información adicional de sensores."

                # --- Prompt 2 Modificado (Veredicto Final Integrado - Estilo Sherlock Holmes) ---
                prompt2 = f"""
Instrucciones: Eres un analista experto en sistemas telemáticos y GPS, aplicando rigurosamente el método deductivo de Sherlock Holmes.

**Información Adicional y Contexto (Importante):**
{gps_info}

Tu objetivo es generar un **"Veredicto Final Integrado" deductivo y fundamentado** sobre el comportamiento de una unidad GPS. Debes **combinar dos fuentes de información clave** (análisis previo y datos históricos) y aplicar un **razonamiento deductivo riguroso** para **eliminar hipótesis menos probables** y **reforzar la hipótesis más plausible** sobre la causa de la interrupción. Utiliza el método de Sherlock Holmes: **deducción lógica, eliminación de imposibilidades y verificación de la hipótesis principal.**

**Manejo de Fechas y Horas:**
- Cuando te refieras a fechas y horas, **NO uses timestamps numéricos**; utiliza un formato legible para humanos, por ejemplo: "25 de Octubre de 2024, 03:15 PM".
- Todas las fechas y horas deben presentarse en la zona horaria **America/Mexico_City**.

**Información para el Análisis Deductivo:**
1. **Análisis Previo (Investigación Inicial):**
{st.session_state['first_analysis']}

2. **Datos Históricos del Período Retroactivo (últimos {retro_period} minutos - Evidencia Adicional):**
{retro_data_text}

**Tu tarea para el "Veredicto Final Integrado" deductivo es la siguiente:**

1. **Resumen Conciso del Análisis Previo (Puntos Clave para la Deducción):**
   Resume brevemente los puntos clave del análisis previo que son **más relevantes para la deducción**, especialmente:
      - Las hipótesis **más probables** identificadas en el análisis previo (priorizadas según la evidencia inicial).
      - Los sensores que mostraron **comportamiento anómalo o tendencias significativas** en el análisis previo (identificando los más "sospechosos").
      - **Cualquier patrón temporal o evento específico** detectado en el análisis previo que pueda ser crucial para la deducción.

2. **Descripción del Desarrollo del Comportamiento en el Período Retroactivo (Análisis Evolutivo Deductivo - Eliminación de Hipótesis):**
   Para cada sensor **relevante y "sospechoso"** identificado en el análisis previo, describe:
      - La **tendencia observada en los datos históricos del período retroactivo**: ¿se mantuvo constante, aumentó, disminuyó, fluctuó de manera errática o mostró algún patrón específico? Describe la tendencia con precisión.
      - **Evalúa si las tendencias observadas en los datos históricos son consistentes o inconsistentes con cada una de las hipótesis planteadas en el análisis previo.** Sé específico:
          * "¿Esta tendencia **refuerza** alguna hipótesis en particular? ¿Por qué?"
          * "¿Esta tendencia **contradice** alguna hipótesis? ¿Cuál? ¿Por qué?"
          * "¿Existen hipótesis que se vuelven **menos probables** a la luz de esta tendencia en los datos históricos? ¿Cuáles?"
      - **Menciona fechas y horas específicas** (en formato "25 de Octubre de 2024, 03:15 PM") si se observan cambios significativos en las tendencias dentro del período retroactivo.
      - **Conecta explícitamente cada tendencia observada con las hipótesis planteadas en el análisis previo, utilizando un razonamiento deductivo claro.**

3. **Veredicto Final Integrado sobre el Comportamiento de la Unidad (Deducción y Conclusión):**
   Emite un **veredicto final claro, conciso y deductivo**, basándote en la **combinación del análisis previo y los datos históricos**, y en el proceso de **eliminación de hipótesis menos probables**. Indica:
      - **¿Cuál es la hipótesis más probable sobre la causa de la interrupción, basándote en la evidencia total? Justifica deductivamente tu conclusión.**
      - ¿Se confirma una falla (interna o de conexión física/externa) y, si es posible, **determina cuándo ocurrió o comenzó a manifestarse** (fecha y hora en formato legible)?
      - Si se identifica un problema potencial que **requiere mayor investigación para confirmar la hipótesis principal**, indica qué tipo de investigación adicional es necesaria y por qué.
      - Si el comportamiento **parece ser normal a pesar de algunas anomalías**, explica por qué se llega a esta conclusión deductivamente, y si las anomalías pueden explicarse por factores externos o fallas en la conexión.
      - Si **no es posible llegar a una conclusión definitiva** incluso con los datos históricos, indica por qué la evidencia no es concluyente y qué datos o información adicional se necesitan imperativamente para llegar a un veredicto deductivo.

4. **Recomendaciones Específicas y Accionables (Verificación de la Hipótesis Principal - Plan de Acción Deductivo):**
   Proporciona una lista de acciones concretas y **deductivamente orientadas** para **verificar el veredicto final y la hipótesis más probable**. Las recomendaciones deben ser específicas, accionables y diseñadas para proporcionar evidencia concluyente a favor o en contra de la hipótesis principal. Incluye:
      - **Inspección física detallada y focalizada** de la unidad GPS, cables, conectores y antena, buscando evidencia que confirme o refute la hipótesis más probable.
      - **Verificación específica** de la firmeza de las conexiones y búsqueda de cables dañados o sueltos en las áreas más relevantes según la hipótesis.
      - **Pruebas de continuidad y resistencia** en los cables específicos (alimentación, antena, datos) que son más relevantes según la hipótesis.
      - **Verificación focalizada** de la alimentación eléctrica en los puntos críticos identificados por la hipótesis.
      - **Diagnóstico interno del GPS** (si es posible y directamente relevante para verificar la hipótesis principal).
      - **Entrevista dirigida al operador** para detectar posibles incidentes o manipulaciones que sean relevantes para la hipótesis principal.
      - **Cualquier otra acción específica y deductivamente justificada** que permita verificar o refutar la hipótesis más probable y llegar a una conclusión definitiva.

**Información Adicional de Sensores:**
{sensor_info}
                """
                if 'chat_session' not in st.session_state:
                    st.session_state['chat_session'] = gemini_model.start_chat(history=[])
                chat_session = st.session_state['chat_session']

                try:
                    response2 = send_message_with_retry(chat_session, prompt2, max_retries=3, delay=5)
                    st.markdown("### Veredicto Final Integrado")
                    st.markdown(response2.text, unsafe_allow_html=True)
                    # Almacenar el segundo análisis en session_state para el Historial de IA
                    st.session_state['second_analysis'] = response2.text
                except google.api_core.exceptions.ResourceExhausted as e:
                    st.error("Se ha excedido la cuota de la API de Gemini (error 429). Intente más tarde o revise su plan.")
                except Exception as e:
                    st.error(f"Ha ocurrido un error al enviar el mensaje: {e}")
