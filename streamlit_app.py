import os
import re
import math
import time
import streamlit as st
import pandas as pd
import folium
from folium.plugins import Fullscreen
from streamlit_folium import folium_static
import plotly.express as px
import datetime
import google.generativeai as genai
import numpy as np
import google.api_core.exceptions

# =============================
#  CONFIGURACIÓN GEMINI AI
# =============================
st.sidebar.header("Configuración de Gemini AI")
gemini_api_key = st.sidebar.text_input("Ingrese su API key de Gemini", type="password")
gemini_enabled = False
if gemini_api_key:
    genai.configure(api_key=gemini_api_key)
    generation_config = {
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 65536,
    }
    gemini_model = genai.GenerativeModel(
        model_name="gemini-2.0-flash",
        generation_config=generation_config,
    )
    gemini_enabled = True
else:
    st.sidebar.warning("Ingrese su API key de Gemini para habilitar el análisis automático.")

# =============================
#  FUNCIONES AUXILIARES
# =============================

def send_message_with_retry(chat_session, message, max_retries=3, delay=60):
    """
    Envía un mensaje a la API de Gemini con reintentos en caso de error 429 (ResourceExhausted).
    """
    for attempt in range(max_retries):
        try:
            response = chat_session.send_message(message)
            return response
        except google.api_core.exceptions.ResourceExhausted:
            st.warning(f"Recurso agotado (429). Reintentando en {delay} segundos... (intento {attempt+1}/{max_retries})")
            time.sleep(delay)
        except Exception as e:
            raise e
    raise google.api_core.exceptions.ResourceExhausted(
        "Se ha agotado el número máximo de reintentos a la API de Gemini."
    )

def haversine(lat1, lon1, lat2, lon2):
    """
    Calcula la distancia en metros entre dos puntos (lat1, lon1) y (lat2, lon2) usando la fórmula haversine.
    """
    R = 6371000  # Radio de la Tierra en metros
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def convert_timestamp(ts):
    """
    Convierte un valor numérico de timestamp (segundos o milisegundos) a un objeto datetime local.
    """
    try:
        ts_val = float(ts)
        if ts_val > 1e11:
            dt = pd.to_datetime(ts_val, unit='ms', utc=True)
        else:
            dt = pd.to_datetime(ts_val, unit='s', utc=True)
        local_tz = "America/Mexico_City"
        dt_local = dt.tz_convert(local_tz).tz_localize(None)
        return dt_local
    except:
        return pd.NaT

def parse_reg_data(uploaded_file):
    """
    Parsea archivo .txt/.wln con líneas tipo: REG;timestamp;lon;lat;io_XX:val,io_YY:val,...
    Retorna un DataFrame con columnas: ['timestamp','longitude','latitude','io_XX',...]
    """
    data = []
    if uploaded_file is not None:
        for line in uploaded_file:
            line = line.decode("utf-8").strip()
            if line.startswith("REG;"):
                parts = line.split(";")
                if len(parts) < 4:
                    continue
                timestamp = parts[1].strip()
                lon = float(parts[2].strip())
                lat = float(parts[3].strip())
                record = {"timestamp": timestamp, "longitude": lon, "latitude": lat}
                for field in parts[4:]:
                    if ":" in field:
                        subfields = field.split(",")
                        for sub in subfields:
                            if ":" in sub:
                                k, v = sub.split(":", 1)
                                k = k.strip().lower()
                                v = v.strip().strip('"')
                                try:
                                    record[k] = float(v)
                                except:
                                    record[k] = v
                data.append(record)
    return pd.DataFrame(data)

def rename_sensor_columns(df, master_mapping):
    """
    Renombra columnas 'io_XX' con nombres y unidades definidos en sensor_master_mapping.
    """
    new_cols = {}
    for col in df.columns:
        if col.startswith("io_"):
            number_part = col[3:].strip()
            if number_part in master_mapping:
                new_cols[col] = master_mapping[number_part]["name"]
    if new_cols:
        df.rename(columns=new_cols, inplace=True)

def calculate_advanced_stats(df):
    """
    Calcula:
    - df['interval_s']: intervalo entre reportes (segundos)
    - df['distance_m']: distancia recorrida entre puntos consecutivos (metros)
    - df['speed_kmh']: velocidad (km/h) aproximada
    Retorna (df, corr_df) donde corr_df es la matriz de correlaciones.
    """
    if df.empty:
        return df, pd.DataFrame()
    df['interval_s'] = df['datetime'].diff().dt.total_seconds()
    df['distance_m'] = np.nan
    for i in range(1, len(df)):
        lat1, lon1 = df.loc[i-1, ['latitude', 'longitude']]
        lat2, lon2 = df.loc[i, ['latitude', 'longitude']]
        dist = haversine(lat1, lon1, lat2, lon2)
        df.loc[i, 'distance_m'] = dist
    df['speed_kmh'] = np.nan
    mask = df['interval_s'] > 0
    df.loc[mask, 'speed_kmh'] = (df.loc[mask, 'distance_m'] / df.loc[mask, 'interval_s']) * 3.6
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_df = pd.DataFrame()
    if len(numeric_cols) > 1:
        corr_df = df[numeric_cols].corr()
    return df, corr_df

def get_pre_termination_data(df, retro_minutes):
    """
    Retorna la ventana retroactiva: subset de df desde (último reporte - retro_minutes) hasta el final.
    También regresa (start_time, last_time).
    """
    last_time = df['datetime'].max()
    start_time = last_time - pd.Timedelta(minutes=retro_minutes)
    subset_df = df[df['datetime'] >= start_time].copy()
    return subset_df, start_time, last_time

def compute_slope(series, timestamps):
    """
    Calcula la pendiente (slope) de la regresión lineal simple de 'series' vs. 'timestamps'.
    """
    if len(series) < 2:
        return np.nan
    t0 = timestamps.min().timestamp()
    x = np.array([t.timestamp() - t0 for t in timestamps])
    return np.polyfit(x, series, 1)[0]

def calcular_tasa_anomalias(pre_term_df, sensor):
    """
    Tasa (%) de valores outliers en pre_term_df[sensor] usando método IQR.
    """
    if sensor not in pre_term_df.columns:
        return 0
    s = pre_term_df[sensor].dropna()
    if s.empty:
        return 0
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    outliers = ((s < (q1 - 1.5 * iqr)) | (s > (q3 + 1.5 * iqr))).sum()
    return round((outliers / len(s)) * 100, 2)

def create_sensor_map(data_points, selected_parameter, opacity,
                      color_min=None, color_max=None, custom_intervals=None,
                      pre_term_indices=None, sensor_master_mapping=None):
    """
    Crea un mapa Folium, coloreando puntos según 'selected_parameter' (numérico o categórico).
    Marca en naranja los puntos del período retroactivo.
    """
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
            val = row[selected_parameter]
            color = get_color(val)
            popup_html = f"<b>{selected_parameter}:</b> {val}<br>"
            io_num = get_io_number(selected_parameter)
            if io_num and sensor_master_mapping and io_num in sensor_master_mapping:
                desc = sensor_master_mapping[io_num].get("description", "")
                popup_html += f"<i>{desc}</i><br>"
            if pre_term_indices is not None and idx in pre_term_indices:
                folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    radius=6, color="orange", fill=True, fill_color="orange",
                    fill_opacity=opacity, 
                    popup=folium.Popup(popup_html, max_width=300)
                ).add_to(sensor_map)
            else:
                folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    radius=5, color=color, fill=True, fill_color=color,
                    fill_opacity=opacity, 
                    popup=folium.Popup(popup_html, max_width=300)
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
        <div style="position: fixed; top: 50px; right: 50px; width: 180px; 
                    border:2px solid grey; z-index:9999; font-size:14px; background-color: white; padding: 10px;">
            <b>Intervalos</b><br>{legend_items}
        </div>
        '''
        sensor_map.get_root().html.add_child(folium.Element(legend_html))
    else:
        if not is_numeric:
            for idx, row in data_points.iterrows():
                val = row[selected_parameter]
                popup_html = f"<b>{selected_parameter}:</b> {val}<br>"
                io_num = get_io_number(selected_parameter)
                if io_num and sensor_master_mapping and io_num in sensor_master_mapping:
                    desc = sensor_master_mapping[io_num].get("description", "")
                    popup_html += f"<i>{desc}</i><br>"
                color = "blue"
                if pre_term_indices and idx in pre_term_indices:
                    color = "orange"
                folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    radius=5, color=color, fill=True, fill_color=color,
                    fill_opacity=opacity, 
                    popup=folium.Popup(popup_html, max_width=300)
                ).add_to(sensor_map)
        else:
            try:
                default_min = data_points[selected_parameter].quantile(0.01)
                default_max = data_points[selected_parameter].quantile(0.99)
            except Exception as e:
                st.error(f"Error calculando rango para {selected_parameter}: {e}")
                return sensor_map
            min_val = color_min if (color_min is not None and color_max is not None and color_min < color_max) else default_min
            max_val = color_max if (color_min is not None and color_max is not None and color_min < color_max) else default_max
            colorscale = px.colors.sequential.Viridis
            for idx, row in data_points.iterrows():
                try:
                    sensor_value = float(row[selected_parameter])
                    if max_val == min_val:
                        color = colorscale[0]
                    else:
                        normalized = (sensor_value - min_val) / (max_val - min_val)
                        normalized = max(0, min(1, normalized))
                        index = int(normalized * (len(colorscale) - 1))
                        color = colorscale[index]
                except:
                    sensor_value = row[selected_parameter]
                    color = "gray"
                popup_html = f"<b>{selected_parameter}:</b> {sensor_value}<br>"
                io_num = get_io_number(selected_parameter)
                if io_num and sensor_master_mapping and io_num in sensor_master_mapping:
                    desc = sensor_master_mapping[io_num].get("description", "")
                    popup_html += f"<i>{desc}</i><br>"
                if pre_term_indices and idx in pre_term_indices:
                    folium.CircleMarker(
                        location=[row['latitude'], row['longitude']],
                        radius=6, color="orange", fill=True, fill_color="orange",
                        fill_opacity=opacity, 
                        popup=folium.Popup(popup_html, max_width=300)
                    ).add_to(sensor_map)
                else:
                    folium.CircleMarker(
                        location=[row['latitude'], row['longitude']],
                        radius=5, color=color, fill=True, fill_color=color,
                        fill_opacity=opacity, 
                        popup=folium.Popup(popup_html, max_width=300)
                    ).add_to(sensor_map)
            stops = 5
            stops_html = ""
            for i in range(stops):
                val_stop = min_val + i*(max_val - min_val)/(stops-1)
                norm = i/(stops-1)
                idx_color = int(norm*(len(colorscale)-1))
                color_stop = colorscale[idx_color]
                stops_html += f'''
                <div style="display:flex; align-items:center;">
                  <div style="background-color:{color_stop}; width:20px; height:10px; margin-right:5px;"></div>
                  {val_stop:.2f}
                </div>
                '''
            legend_html = f'''
            <div style="position: fixed; top: 50px; right: 50px; width: 150px; 
                        border:2px solid grey; z-index:9999; font-size:14px; background-color: white; padding: 10px;">
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
    """
    Crea un mapa básico solo con la ubicación (sin coloreo por sensor específico).
    """
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
        color = "blue"
        if pre_term_indices and idx in pre_term_indices:
            color = "orange"
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=5, color=color, fill=True, fill_color=color,
            fill_opacity=opacity, 
            popup=folium.Popup(popup_html, max_width=300)
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
    Convierte texto (Markdown) a PDF usando xhtml2pdf sin requerir la librería 'markdown'.
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

def build_advanced_prompt(df, stats_pre_df, corr_df, sensor_master_df, gps_info, datos_completos="",
                          incluir_info_gps=True, incluir_metricas=True,
                          incluir_estadisticas=True, incluir_datos_historicos=False):
    """
    Construye el prompt para la API de Gemini, incluyendo (o no) las secciones según lo seleccionado por el usuario:
    1. Información de GPS y lista de sensores.
    2. Métricas avanzadas (intervalos, velocidad y correlaciones).
    3. Estadísticas retroactivas de cada sensor.
    4. Datos históricos completos.
    
    Se instruye a la IA a analizar individualmente cada aspecto y a integrar los hallazgos usando el método científico.
    """
    # Lista de sensores
    sensors_text = ""
    if sensor_master_df is not None:
        sensors_text = "Listado de Sensores (Sensor Master List):\n"
        for idx, row in sensor_master_df.iterrows():
            property_id = str(row["Property ID in AVL packet"]).strip()
            name = row["Property Name"]
            units = row["Units"] if pd.notna(row["Units"]) else "N/A"
            desc = row["Description"] if pd.notna(row["Description"]) else ""
            sensors_text += f"- io_{property_id} | {name} | {units} | {desc}\n"
    else:
        sensors_text = "Sensor Master List no cargada.\n"

    # Métricas avanzadas
    interval_summary = ""
    if "interval_s" in df.columns:
        mean_interval = df["interval_s"].mean()
        max_interval  = df["interval_s"].max()
        interval_summary += f"- Intervalo promedio entre reportes: {mean_interval:.2f} seg\n"
        interval_summary += f"- Intervalo máximo entre reportes: {max_interval:.2f} seg\n"
    if "speed_kmh" in df.columns:
        avg_speed = df["speed_kmh"].mean(skipna=True)
        interval_summary += f"- Velocidad promedio: {avg_speed:.2f} km/h\n"
    top_corr_text = ""
    if not corr_df.empty:
        corr_unstack = corr_df.unstack().reset_index()
        corr_unstack.columns = ["Sensor1", "Sensor2", "Correlation"]
        corr_unstack = corr_unstack[corr_unstack["Sensor1"] != corr_unstack["Sensor2"]]
        corr_unstack["abs_corr"] = corr_unstack["Correlation"].abs()
        corr_unstack = corr_unstack.sort_values("abs_corr", ascending=False)
        top_corr = corr_unstack.head(5)
        top_corr_text = "Principales correlaciones entre sensores:\n"
        for _, row_ in top_corr.iterrows():
            top_corr_text += f"  - {row_['Sensor1']} vs {row_['Sensor2']}: {row_['Correlation']:.2f}\n"

    # Estadísticas retroactivas
    stats_text = ""
    if not stats_pre_df.empty:
        stats_text = "Estadísticas Retroactivas por Sensor:\n"
        for i, r in stats_pre_df.iterrows():
            sensor = r["Sensor"]
            prom_total = r["Promedio Total"]
            prom_retro = r["Promedio Retroactivo"]
            tasa_anom  = r["Tasa Anomalías Retro (%)"]
            stats_text += (f"- {sensor}: Promedio (Total={prom_total:.2f}, Retro={prom_retro:.2f}), "
                           f"Tasa de Anomalías Retro={tasa_anom:.2f}%\n")

    info_gps_section = (f"**Información de GPS y Sensor Master List:**\n{gps_info}\n\n{sensors_text}\n" 
                        if incluir_info_gps else "Información de GPS y Sensor Master List omitida.\n")
    metricas_section = (f"**Métricas Avanzadas (Intervalos, Velocidad, Correlaciones):**\n{interval_summary}\n{top_corr_text}\n" 
                        if incluir_metricas else "Métricas Avanzadas omitidas.\n")
    estadisticas_section = (f"**Estadísticas Retroactivas por Sensor:**\n{stats_text}\n" 
                            if incluir_estadisticas else "Estadísticas Retroactivas omitidas.\n")
    datos_historicos_section = (f"**Datos Históricos Completos:**\n{datos_completos}\n" 
                                if incluir_datos_historicos else "Datos Históricos omitidos.\n")

    prompt = f"""
Eres un analista experto en sistemas telemáticos y GPS, con amplio conocimiento del método científico.
Realiza un análisis detallado y estructurado de la siguiente información siguiendo estos pasos:

1. **Observación y Descripción**: Revisa la información proporcionada, identificando los sensores relevantes y sus características.
2. **Análisis de Métricas Avanzadas**: Evalúa los intervalos entre reportes, la velocidad calculada y las correlaciones entre sensores, señalando comportamientos anómalos.
3. **Evaluación de Estadísticas Retroactivas**: Compara las métricas totales con las retroactivas para detectar variaciones significativas en cada sensor.
4. **Revisión de Datos Históricos**: (Si se han incluido) Examina los datos históricos en busca de patrones o tendencias.
5. **Integración y Conclusión**: Con base en lo anterior, formula hipótesis sobre las posibles causas de fallos o anomalías (ej. problemas de alimentación, interferencia, fallos en sensores) y concluye de forma coherente.

A continuación, se detalla la información disponible:

{info_gps_section}

{metricas_section}

{estadisticas_section}

{datos_historicos_section}

Realiza el análisis siguiendo estos pasos y proporciona una conclusión final integradora.
    
Comienza tu análisis ahora:
"""
    return prompt

# ==================================================
#    MAIN APP / ESTRUCTURA EN PESTAÑAS
# ==================================================
def main():
    # 1. Carga de la Sensor Master List
    st.sidebar.header("Sensor Master List")
    master_file = st.sidebar.file_uploader("Cargar Sensor Master List (CSV o Excel)", 
                                           type=["csv", "xls", "xlsx"], 
                                           key="master_file")
    
    sensor_master_df = None
    sensor_master_mapping = {}
    if master_file is not None:
        file_name = master_file.name.lower()
        try:
            if file_name.endswith(".csv"):
                sensor_master_df = pd.read_csv(master_file)
                sensor_master_df["Sheet"] = "CSV"
            else:
                sheets_dict = pd.read_excel(master_file, sheet_name=None)
                sensor_master_df = pd.concat(
                    [df_.assign(Sheet=sh) for sh, df_ in sheets_dict.items()],
                    ignore_index=True
                )
        except Exception as e:
            st.sidebar.error(f"Error al leer el archivo: {e}")
        if sensor_master_df is not None:
            required_cols = ["Property ID in AVL packet", "Property Name", "Units", "Description"]
            missing = [c for c in required_cols if c not in sensor_master_df.columns]
            if missing:
                st.sidebar.error(f"Faltan columnas: {missing}")
            else:
                st.sidebar.success("Sensor Master List cargada correctamente.")
                for idx, row in sensor_master_df.iterrows():
                    key = str(row["Property ID in AVL packet"]).strip()
                    p_name = row["Property Name"]
                    units_ = str(row["Units"]) if not pd.isna(row["Units"]) else ""
                    desc_  = row["Description"] if not pd.isna(row["Description"]) else ""
                    if units_.strip():
                        full_name = f"{p_name} (io_{key}) [{units_}]"
                    else:
                        full_name = f"{p_name} (io_{key})"
                    sensor_master_mapping[key] = {
                        "name": full_name,
                        "description": desc_
                    }
    with st.sidebar.expander("Ver Sensor Master List"):
        if sensor_master_df is not None:
            st.dataframe(sensor_master_df)
        else:
            st.write("No se ha cargado el archivo de Sensor Master List.")

    # 2. Configuración de análisis retroactivo
    st.sidebar.header("Análisis Retroactivo")
    retro_period = st.sidebar.number_input("Periodo de análisis retroactivo (min)", 
                                           min_value=1, value=15, step=1)

    # 3. Información adicional de GPS
    st.sidebar.header("Información Adicional de GPS")
    gps_info = st.sidebar.text_area("Proporciona info extra (fallas, contexto, etc.)")

    # 4. Pestañas principales
    tabs = st.tabs(["Análisis", "Conversación", "Historial IA"])
    with tabs[0]:
        st.title("Sensor Insights - Análisis del Corte de Reporte")
        uploaded_file = st.file_uploader("Cargar archivo de datos (TXT/WLN)", 
                                         type=["txt", "wln"], 
                                         key="data_file")
        if uploaded_file:
            if "data_file_obj" not in st.session_state:
                st.session_state["data_file_obj"] = uploaded_file
            df = parse_reg_data(uploaded_file)
            if df.empty:
                st.error("No se pudieron procesar los datos. Revisa el formato REG.")
                st.stop()
            if "timestamp" in df.columns:
                df["datetime"] = df["timestamp"].apply(convert_timestamp)
                st.write("Rango de fechas en los datos:", df["datetime"].min(), "a", df["datetime"].max())
                last_dt = df["datetime"].max()
                st.markdown(f"### Último mensaje: **{last_dt}**")
                date_range = st.date_input("Seleccione rango de fechas",
                                           [df["datetime"].min().date(), df["datetime"].max().date()])
                if isinstance(date_range, list) and len(date_range) == 2:
                    d1, d2 = date_range
                    df = df[(df["datetime"].dt.date >= d1) & (df["datetime"].dt.date <= d2)]
                    if df.empty:
                        st.error("No hay datos en ese rango.")
                        st.stop()
                st.write("Filtrar por hora:")
                tstart = st.time_input("Hora inicio", value=datetime.time(0,0))
                tend   = st.time_input("Hora fin", value=datetime.time(23,59))
                df = df[(df["datetime"].dt.time >= tstart) & (df["datetime"].dt.time <= tend)]
                if df.empty:
                    st.error("No hay datos en ese rango horario.")
                    st.stop()
            else:
                st.error("No existe la columna 'timestamp' en el archivo.")
                st.stop()
            if sensor_master_mapping:
                rename_sensor_columns(df, sensor_master_mapping)
            df = df.sort_values("datetime").reset_index(drop=True)
            df, corr_df = calculate_advanced_stats(df)
            pre_term_df, retro_start, last_time = get_pre_termination_data(df, retro_period)
            pre_term_indices = pre_term_df.index.tolist()
            st.subheader("Análisis Retroactivo")
            st.write(f"Analizando datos desde {retro_start} hasta {last_time}")
            st.dataframe(pre_term_df.head())
            available_columns = [c for c in df.columns if c not in ["latitude", "longitude", "timestamp", "datetime"]]
            sensor_stats = []
            for col in available_columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    overall_mean   = df[col].mean()
                    overall_median = df[col].median()
                    overall_std    = df[col].std()
                    overall_min    = df[col].min()
                    overall_max    = df[col].max()
                    overall_p5     = df[col].quantile(0.05)
                    overall_p95    = df[col].quantile(0.95)
                    pre_mean   = pre_term_df[col].mean()
                    pre_median = pre_term_df[col].median()
                    pre_std    = pre_term_df[col].std()
                    pre_min    = pre_term_df[col].min()
                    pre_max    = pre_term_df[col].max()
                    pre_p5     = pre_term_df[col].quantile(0.05)
                    pre_p95    = pre_term_df[col].quantile(0.95)
                    diff_mean   = overall_mean - pre_mean
                    diff_median = overall_median - pre_median
                    pct_diff_mean   = (diff_mean/overall_mean*100) if overall_mean != 0 else 0
                    pct_diff_median = (diff_median/overall_median*100) if overall_median != 0 else 0
                    all_slope = compute_slope(df[col].dropna(), df["datetime"][df[col].notna()])
                    r_slope   = compute_slope(pre_term_df[col].dropna(), pre_term_df["datetime"][pre_term_df[col].notna()])
                    diff_slope= all_slope - r_slope if not (np.isnan(all_slope) or np.isnan(r_slope)) else np.nan
                    q1 = df[col].quantile(0.25)
                    q3 = df[col].quantile(0.75)
                    iqr = q3 - q1
                    total_outliers = ((df[col] < q1-1.5*iqr) | (df[col] > q3+1.5*iqr)).sum()
                    pre_outliers   = ((pre_term_df[col] < q1-1.5*iqr) | (pre_term_df[col] > q3+1.5*iqr)).sum()
                    tasa_anom = calcular_tasa_anomalias(pre_term_df, col)
                    sensor_stats.append({
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
                        "Pendiente Total": all_slope,
                        "Pendiente Retroactivo": r_slope,
                        "Diferencia de Pendiente": diff_slope,
                        "Outliers Totales": total_outliers,
                        "Outliers Retroactivos": pre_outliers,
                        "Tasa Anomalías Retro (%)": tasa_anom
                    })
            stats_pre_df = pd.DataFrame(sensor_stats)
            st.subheader("Estadísticas Mejoradas de Sensores en el Período Retroactivo")
            st.dataframe(stats_pre_df)
            opacity = st.slider("Opacidad de los marcadores", 0.1, 1.0, 0.7)
            custom_intervals = None
            color_min = None
            color_max = None
            selected_parameter = None
            if available_columns:
                selected_parameter = st.selectbox("Selecciona el sensor para visualizar:", available_columns)
                if pd.api.types.is_numeric_dtype(df[selected_parameter]):
                    sensor_min = df[selected_parameter].min()
                    sensor_max = df[selected_parameter].max()
                    st.write("Rango de valores para", selected_parameter, ":", sensor_min, "a", sensor_max)
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
                            c1, c2, c3, c4 = st.columns([1, 3, 3, 3])
                            enabled = c1.checkbox(f"Activar Intervalo {i+1}", value=True, key=f"interval_{i}")
                            with c2:
                                min_val = st.number_input(f"Min (Intervalo {i+1})", key=f"min_{i}", value=float(default_min))
                            with c3:
                                max_val = st.number_input(f"Max (Intervalo {i+1})", key=f"max_{i}", value=float(default_max))
                            with c4:
                                color_val = st.color_picker(f"Color (Intervalo {i+1})", key=f"color_{i}", value="#0000ff")
                            if enabled:
                                custom_intervals.append({"min": min_val, "max": max_val, "color": color_val})
                else:
                    selected_parameter = selected_parameter
            else:
                st.info("No hay columnas de sensor para visualizar. Se mostrará mapa básico.")
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
                    fig_hist = px.histogram(df, x=selected_parameter, nbins=50, marginal="box", color_discrete_sequence=['#636EFA'])
                    fig_hist.update_layout(xaxis_title=selected_parameter, yaxis_title="Frecuencia", bargap=0.1)
                    st.plotly_chart(fig_hist)
                except Exception as e:
                    st.error(f"Error creando histograma: {e}")
            if st.checkbox("Mostrar datos en tabla"):
                st.subheader("Datos Completos")
                st.dataframe(df)
            csv_data = df.to_csv(index=False).encode('utf-8')
            st.download_button("Descargar datos en CSV", csv_data, "datos.csv", "text/csv", key="download_csv")
            if not corr_df.empty:
                st.subheader("Matriz de Correlación")
                st.dataframe(corr_df)
                fig_corr = px.imshow(corr_df, text_auto=True, aspect="auto", color_continuous_scale='RdBu', origin='lower')
                st.plotly_chart(fig_corr)
                with st.expander("¿Cómo interpretar la Matriz de Correlación?"):
                    st.markdown("""
La **matriz de correlación** muestra el grado de relación lineal entre cada par de variables numéricas, con valores entre **-1** y **1**.

- **1** indica una correlación perfecta positiva: a mayor valor de una variable, mayor el valor de la otra.
- **-1** indica una correlación perfecta negativa: a mayor valor de una variable, menor el valor de la otra.
- **0** indica ausencia de relación lineal.

**Ejemplo:**
- Si la correlación entre **Sensor A** y **Sensor B** es **0.85**, significa que cuando el Sensor A aumenta, el Sensor B también tiende a aumentar de manera fuerte.
- Si la correlación entre **Sensor C** y **Sensor D** es **-0.5**, indica una relación negativa moderada, es decir, a mayor Sensor C, menor Sensor D.

Esta herramienta es útil para identificar relaciones y posibles redundancias entre variables.
                    """)
            st.markdown("### Seleccionar secciones a incluir en el análisis automático")
            incluir_info_gps = st.checkbox("Incluir Información de GPS y Sensor Master List", value=True)
            incluir_metricas = st.checkbox("Incluir Métricas Avanzadas (intervalos, velocidad y correlaciones)", value=True)
            incluir_estadisticas = st.checkbox("Incluir Estadísticas Retroactivas de cada sensor", value=True)
            incluir_datos_historicos = st.checkbox("Incluir Datos Históricos Completos", value=False)
            
            if gemini_enabled:
                if st.button("Obtener Análisis Automático con Gemini AI"):
                    datos_completos = ""
                    if incluir_datos_historicos:
                        datos_completos = "\n\nDatos Completos:\n" + df.to_csv(index=False)
                    prompt_optimizado = build_advanced_prompt(
                        df=df, 
                        stats_pre_df=stats_pre_df,
                        corr_df=corr_df,
                        sensor_master_df=sensor_master_df,
                        gps_info=gps_info,
                        datos_completos=datos_completos,
                        incluir_info_gps=incluir_info_gps,
                        incluir_metricas=incluir_metricas,
                        incluir_estadisticas=incluir_estadisticas,
                        incluir_datos_historicos=incluir_datos_historicos
                    )
                    if "chat_session" not in st.session_state:
                        st.session_state["chat_session"] = gemini_model.start_chat(history=[])
                    chat_session = st.session_state["chat_session"]
                    try:
                        response = send_message_with_retry(chat_session, prompt_optimizado, max_retries=3, delay=60)
                        analysis_text = response.text
                        st.subheader("Análisis Automático Gemini (Optimizado)")
                        st.markdown(analysis_text, unsafe_allow_html=True)
                        st.session_state["first_analysis"] = analysis_text
                        pdf_bytes = convert_markdown_to_pdf(analysis_text)
                        if pdf_bytes:
                            st.download_button("Descargar análisis a PDF", data=pdf_bytes, file_name="analisis_gemini.pdf", mime="application/pdf")
                    except google.api_core.exceptions.ResourceExhausted:
                        st.error("Se excedió la cuota de la API (error 429).")
                    except Exception as e:
                        st.error(f"Error enviando mensaje: {e}")

    with tabs[1]:
        st.subheader("Conversación Iterativa con Gemini AI")
        conv_context = ""
        if "data_file_obj" in st.session_state:
            st.session_state.data_file_obj.seek(0)
            df_conv = parse_reg_data(st.session_state.data_file_obj)
            if not df_conv.empty and "timestamp" in df_conv.columns:
                df_conv["datetime"] = df_conv["timestamp"].apply(convert_timestamp)
                conv_start_date = st.date_input("Fecha inicio (conv)", value=df_conv["datetime"].min().date(), key="conv_start_date")
                conv_start_time = st.time_input("Hora inicio (conv)", value=datetime.time(0,0), key="conv_start_time")
                conv_end_date   = st.date_input("Fecha fin (conv)", value=df_conv["datetime"].max().date(), key="conv_end_date")
                conv_end_time   = st.time_input("Hora fin (conv)", value=datetime.time(23,59), key="conv_end_time")
                start_dt = datetime.datetime.combine(conv_start_date, conv_start_time)
                end_dt   = datetime.datetime.combine(conv_end_date, conv_end_time)
                df_conv  = df_conv[(df_conv["datetime"] >= start_dt) & (df_conv["datetime"] <= end_dt)]
                df_conv["fecha_formateada"] = df_conv["datetime"].apply(lambda dt: dt.strftime("%d de %B de %Y, %I:%M:%S %p"))
                df_conv_context = df_conv.drop(columns=["timestamp", "datetime"], errors="ignore")
                conv_context = "Datos filtrados:\n" + df_conv_context.to_csv(index=False) + "\n"
        extra_context = ("***REGLAS***:\n- Usa formateo legible para fechas.\n- Zona horaria: America/Mexico_City.\n\n")
        include_file_context = st.checkbox("Incluir contexto de Sensor Master List y datos", value=True)
        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []
        user_input = st.text_input("Mensaje:")
        if st.button("Enviar"):
            if user_input:
                context_text = extra_context
                if include_file_context:
                    if sensor_master_df is not None:
                        context_text += "Sensor Master:\n" + sensor_master_df.to_csv(index=False) + "\n"
                    context_text += conv_context
                final_message = user_input + "\n\n" + context_text
                if "chat_session" not in st.session_state:
                    st.session_state["chat_session"] = gemini_model.start_chat(history=[])
                chat_session = st.session_state["chat_session"]
                try:
                    response = send_message_with_retry(chat_session, final_message)
                    st.session_state["chat_history"].append({"user": user_input, "ai": response.text})
                except google.api_core.exceptions.ResourceExhausted:
                    st.error("Se excedió la cuota de la API (429).")
                except Exception as e:
                    st.error(f"Error: {e}")
        if st.session_state.get("chat_history"):
            st.markdown("### Historial")
            for exchange in st.session_state["chat_history"]:
                st.markdown(f"**Tú:** {exchange['user']}")
                st.markdown(f"**AI:** {exchange['ai']}")
    with tabs[2]:
        st.subheader("Historial de Análisis de Gemini AI")
        if "first_analysis" in st.session_state:
            st.markdown("#### Análisis Automático (Primer Análisis)")
            st.markdown(st.session_state["first_analysis"], unsafe_allow_html=True)
        else:
            st.info("Aún no se ha realizado el análisis automático.")
        st.markdown("---")
        if "second_analysis" in st.session_state:
            st.markdown("#### Veredicto Final Integrado (Segundo Análisis)")
            st.markdown(st.session_state["second_analysis"], unsafe_allow_html=True)
        else:
            st.info("Aún no se ha ejecutado el veredicto final.")
        if gemini_enabled:
            if st.button("Obtener Veredicto Final Integrado"):
                if "first_analysis" not in st.session_state:
                    st.error("Primero obtén el Análisis Automático.")
                else:
                    prompt2 = f"""
Basado en el análisis previo y los datos retroactivos, genera un veredicto final que integre todos los hallazgos y siga el método científico (observación, hipótesis, experimentación y conclusión). Especifica los sensores relevantes y las métricas analizadas.
"""
                    if "chat_session" not in st.session_state:
                        st.session_state["chat_session"] = gemini_model.start_chat(history=[])
                    chat_session = st.session_state["chat_session"]
                    try:
                        response2 = send_message_with_retry(chat_session, prompt2)
                        st.session_state["second_analysis"] = response2.text
                        st.markdown("### Veredicto Final Integrado")
                        st.markdown(response2.text, unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
