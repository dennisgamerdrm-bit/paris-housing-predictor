# frontend/app.py
import os
import streamlit as st
import requests
import pandas as pd
import json
from datetime import datetime

# ------------------------------------------------------------
# 1. CONFIGURACIÓN DE LA PÁGINA
# ------------------------------------------------------------
st.set_page_config(
    page_title="Tasador IA de París Pro",
    page_icon="🏠",
    layout="wide"
)

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")


# ------------------------------------------------------------
# 2. SIDEBAR CON ESTADÍSTICAS (Opción B)
# ------------------------------------------------------------
with st.sidebar:
    st.header("📊 Estadísticas de Uso")
    
    try:
        response = requests.get(f"{BACKEND_URL}/stats", timeout=5)
        if response.status_code == 200:
            stats = response.json()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Predicciones", stats['total_predicciones'])
            with col2:
                st.metric("Precio Promedio", f"{stats['precio_promedio']:,.0f} €" if stats['precio_promedio'] else "N/A")
            
            if stats['distrito_mas_consultado']:
                st.info(f"📍 Distrito más consultado: **{stats['distrito_mas_consultado']}**")
            
            if stats['ultima_prediccion']:
                fecha = datetime.fromisoformat(stats['ultima_prediccion'])
                st.caption(f"Última consulta: {fecha.strftime('%d/%m/%Y %H:%M')}")
        else:
            st.warning("No se pudieron cargar estadísticas")
    except:
        st.error("Backend no disponible")
    
    st.divider()
    
    st.header("ℹ️ Acerca de")
    st.markdown("""
    **Modelo:** Random Forest  
    **Precisión (R²):** 76.6%  
    **Error Medio:** ±332k €
    
    ---
    **Stack:**
    - 🐍 Python
    - ⚡ FastAPI
    - 🎈 Streamlit
    - 💾 SQLite
    - 🐳 Docker
    """)

# ------------------------------------------------------------
# 3. PÁGINA PRINCIPAL
# ------------------------------------------------------------
st.title("🏠 Tasador de Viviendas en París Pro")
st.markdown("*Powered by IA + Historial en Base de Datos*")

# Tabs para organizar
tab1, tab2 = st.tabs(["🏢 Predicción Individual", "📁 Procesamiento Batch (CSV)"])

# ------------------------------------------------------------
# TAB 1: PREDICCIÓN INDIVIDUAL
# ------------------------------------------------------------
with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        distrito = st.selectbox("📍 Distrito", options=list(range(1, 21)))
        metros = st.slider("📐 m²", 15, 200, 80)
        habitaciones = st.slider("🛏️ Habitaciones", 1, 6, 3)
        planta = st.slider("🏢 Planta", 0, 10, 3)
    
    with col2:
        year_built = st.slider("📅 Año", 1850, 2023, 1980)
        distancia = st.slider("🚶 Km al centro", 0.5, 15.0, 5.0)
        tipo = st.selectbox("🏢 Tipo", ["Apartment", "Studio", "Loft", "Penthouse"])
        condicion = st.selectbox("🔧 Estado", ["New", "Good", "Renovated", "Needs Renovation"])
    
    if st.button("💰 Valorar propiedad", type="primary"):
        payload = {
            "Arrondissement": distrito, "Size_sqm": metros, "Rooms": habitaciones,
            "Floor": planta, "Year_Built": year_built, "Distance_to_Center_km": distancia,
            "Property_Type": tipo, "Condition": condicion
        }
        
        with st.spinner("🧠 Analizando..."):
            try:
                response = requests.post(f"{BACKEND_URL}/predict", json=payload, timeout=10)
                if response.status_code == 200:
                    res = response.json()
                    st.success("✅ ¡Valoración completada!")
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("💰 Estimado", f"{res['precio_estimado']:,.0f} €")
                    col2.metric("📊 Mínimo", f"{res['rango_inferior']:,.0f} €")
                    col3.metric("📈 Máximo", f"{res['rango_superior']:,.0f} €")
                else:
                    st.error(f"Error: {response.text}")
            except Exception as e:
                st.error(f"No se pudo conectar con el backend: {e}")

# ------------------------------------------------------------
# TAB 2: PROCESAMIENTO BATCH (Opción A)
# ------------------------------------------------------------
with tab2:
    st.subheader("📁 Sube un archivo CSV con múltiples propiedades")
    st.markdown("""
    El archivo debe contener las columnas:  
    `Arrondissement, Size_sqm, Rooms, Floor, Year_Built, Distance_to_Center_km, Property_Type, Condition`
    """)
    
    archivo = st.file_uploader("Selecciona tu CSV", type=["csv"])
    
    if archivo:
        df = pd.read_csv(archivo)
        st.write(f"📋 **{len(df)} propiedades cargadas**")
        st.dataframe(df.head())
        
        if st.button("🚀 Procesar lote completo", type="primary"):
            files = {"file": (archivo.name, archivo.getvalue(), "text/csv")}
            
            with st.spinner(f"Procesando {len(df)} propiedades..."):
                try:
                    response = requests.post(f"{BACKEND_URL}/predict_batch", files=files, timeout=120)
                    
                    if response.status_code == 200:
                        res = response.json()
                        st.success(f"✅ {res['exitosos']} de {res['total_procesados']} procesados correctamente")
                        
                        df_result = pd.DataFrame(res['data'])
                        st.dataframe(df_result)
                        
                        csv = df_result.to_csv(index=False)
                        st.download_button(
                            "⬇️ Descargar CSV con predicciones",
                            csv,
                            "predicciones_paris.csv",
                            "text/csv"
                        )
                    else:
                        st.error(f"Error: {response.text}")
                except Exception as e:
                    st.error(f"Error de conexión: {e}")