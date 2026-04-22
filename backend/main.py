# backend/main.py
import joblib
import pandas as pd
import os
import json
import time
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import aiosqlite
from contextlib import asynccontextmanager

# ------------------------------------------------------------
# 1. CONFIGURACIÓN INICIAL (Cargar el cerebro congelado)
# ------------------------------------------------------------
print("🚀 Iniciando servidor de IA...")

# Rutas configurables por variables de entorno (para compatibilidad con Docker)
MODELOS_PATH = os.getenv("MODELOS_PATH", "../modelos_entrenados")
DB_PATH = os.getenv("DB_PATH", "../predicciones.db")

# Construir rutas completas
RUTA_MODELO = os.path.join(MODELOS_PATH, "modelo_precios_paris.pkl")
RUTA_MAPEO = os.path.join(MODELOS_PATH, "mapeo_condition.pkl")
RUTA_COLUMNAS = os.path.join(MODELOS_PATH, "columnas_input.pkl")

# Asegurar que el directorio de la DB existe
os.makedirs(os.path.dirname(DB_PATH) if os.path.dirname(DB_PATH) else ".", exist_ok=True)

print(f"📁 Ruta de modelos: {MODELOS_PATH}")
print(f"💾 Base de datos: {DB_PATH}")

try:
    MODELO = joblib.load(RUTA_MODELO)
    MAPEO_CONDITION = joblib.load(RUTA_MAPEO)
    COLUMNAS_ESPERADAS = joblib.load(RUTA_COLUMNAS)
    print("✅ Modelo y reglas cargados en memoria.")
    print(f"   Columnas esperadas: {len(COLUMNAS_ESPERADAS)}")
except Exception as e:
    print(f"❌ ERROR FATAL: No se pudo cargar el modelo. {e}")
    print(f"   Verifica que existen:")
    print(f"   - {RUTA_MODELO}")
    print(f"   - {RUTA_MAPEO}")
    print(f"   - {RUTA_COLUMNAS}")
    exit()

# ------------------------------------------------------------
# 2. INICIALIZAR BASE DE DATOS SQLITE
# ------------------------------------------------------------
async def init_db():
    """Inicializa la base de datos SQLite"""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute('''
            CREATE TABLE IF NOT EXISTS predicciones (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                distrito INTEGER,
                metros INTEGER,
                habitaciones INTEGER,
                planta INTEGER,
                year_built INTEGER,
                distancia_centro REAL,
                tipo_propiedad TEXT,
                condicion TEXT,
                precio_predicho REAL,
                tiempo_respuesta_ms REAL
            )
        ''')
        await db.commit()
    print(f"✅ Base de datos SQLite inicializada en {DB_PATH}")

async def guardar_prediccion(datos: dict, precio: float, tiempo_ms: float):
    """Guarda una predicción en la base de datos"""
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute('''
                INSERT INTO predicciones 
                (timestamp, distrito, metros, habitaciones, planta, year_built, 
                 distancia_centro, tipo_propiedad, condicion, precio_predicho, tiempo_respuesta_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                datos.get('Arrondissement'),
                datos.get('Size_sqm'),
                datos.get('Rooms'),
                datos.get('Floor'),
                datos.get('Year_Built'),
                datos.get('Distance_to_Center_km'),
                datos.get('Property_Type'),
                datos.get('Condition'),
                precio,
                tiempo_ms
            ))
            await db.commit()
    except Exception as e:
        print(f"⚠️ Error guardando en DB: {e}")

# ------------------------------------------------------------
# 3. MODELOS DE DATOS (Pydantic)
# ------------------------------------------------------------
class DatosCasa(BaseModel):
    Arrondissement: int
    Size_sqm: int
    Rooms: int
    Floor: int
    Year_Built: int
    Distance_to_Center_km: float
    Property_Type: str
    Condition: str

class PrediccionResponse(BaseModel):
    precio_estimado: float
    moneda: str
    rango_inferior: float
    rango_superior: float

class StatsResponse(BaseModel):
    total_predicciones: int
    precio_promedio: float
    ultima_prediccion: Optional[str] = None
    distrito_mas_consultado: Optional[int] = None

# ------------------------------------------------------------
# 4. CREAR LA APLICACIÓN FASTAPI (con Lifespan moderno)
# ------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Maneja el ciclo de vida de la aplicación (reemplaza a on_event)"""
    print("🔵 Iniciando aplicación...")
    await init_db()
    yield
    print("🔴 Cerrando aplicación...")

app = FastAPI(
    title="Paris Housing Price Predictor Pro",
    description="IA con historial y procesamiento batch",
    version="2.0.0",
    lifespan=lifespan
)

# ------------------------------------------------------------
# 5. FUNCIÓN DE LIMPIEZA REUTILIZABLE
# ------------------------------------------------------------
def limpiar_y_predecir(datos: dict) -> float:
    """Aplica la limpieza y devuelve la predicción"""
    df_input = pd.DataFrame([datos])
    
    # One-Hot Encoding para Property_Type
    dummies = pd.get_dummies(df_input['Property_Type'], prefix='Property_Type')
    df_limpio = pd.concat([df_input, dummies], axis=1)
    df_limpio = df_limpio.drop(columns=['Property_Type'])
    
    # Mapeo Ordinal para Condition
    df_limpio['Condition_encoded'] = df_limpio['Condition'].map(MAPEO_CONDITION).fillna(0)
    df_limpio = df_limpio.drop(columns=['Condition'])
    
    # Alinear columnas
    df_final = df_limpio.reindex(columns=COLUMNAS_ESPERADAS, fill_value=0)
    
    # Predecir
    return MODELO.predict(df_final)[0]

# ------------------------------------------------------------
# 6. ENDPOINT PRINCIPAL (/predict)
# ------------------------------------------------------------
@app.post("/predict", response_model=PrediccionResponse)
async def predecir_precio(datos: DatosCasa):
    """Predice el precio de UNA propiedad"""
    inicio = time.time()
    
    try:
        input_dict = datos.model_dump()
        prediccion = limpiar_y_predecir(input_dict)
        tiempo_ms = (time.time() - inicio) * 1000
        
        # Guardar en DB
        await guardar_prediccion(input_dict, prediccion, tiempo_ms)
        
        return {
            "precio_estimado": round(prediccion, 2),
            "moneda": "EUR",
            "rango_inferior": round(prediccion * 0.78, 2),
            "rango_superior": round(prediccion * 1.22, 2)
        }
        
    except Exception as e:
        print(f"🔥 Error en /predict: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ------------------------------------------------------------
# 7. ENDPOINT BATCH (/predict_batch)
# ------------------------------------------------------------
@app.post("/predict_batch")
async def predecir_lote(file: UploadFile = File(...)):
    """Procesa un archivo CSV con múltiples propiedades"""
    inicio_total = time.time()
    
    try:
        # Leer CSV
        contents = await file.read()
        df = pd.read_csv(pd.io.common.BytesIO(contents))
        
        print(f"📊 Procesando lote de {len(df)} propiedades...")
        
        resultados = []
        tiempos = []
        
        for idx, row in df.iterrows():
            inicio_fila = time.time()
            try:
                # Construir payload
                datos = {
                    'Arrondissement': int(row['Arrondissement']),
                    'Size_sqm': int(row['Size_sqm']),
                    'Rooms': int(row['Rooms']),
                    'Floor': int(row['Floor']),
                    'Year_Built': int(row['Year_Built']),
                    'Distance_to_Center_km': float(row['Distance_to_Center_km']),
                    'Property_Type': row['Property_Type'],
                    'Condition': row['Condition']
                }
                
                # Predecir
                pred = limpiar_y_predecir(datos)
                resultados.append(pred)
                
                tiempo_ms = (time.time() - inicio_fila) * 1000
                tiempos.append(tiempo_ms)
                
                # Guardar en DB
                await guardar_prediccion(datos, pred, tiempo_ms)
                
            except Exception as e:
                print(f"⚠️ Error en fila {idx}: {e}")
                resultados.append(None)
                tiempos.append(None)
        
        df['Precio_Predicho'] = resultados
        tiempo_total = (time.time() - inicio_total) * 1000
        
        print(f"✅ Lote completado: {sum(1 for x in resultados if x is not None)}/{len(resultados)} exitosos en {tiempo_total:.0f}ms")
        
        return {
            "total_procesados": len(resultados),
            "exitosos": sum(1 for x in resultados if x is not None),
            "data": df.to_dict(orient='records'),
            "tiempo_promedio_ms": sum(t for t in tiempos if t) / len([t for t in tiempos if t]) if tiempos else 0,
            "tiempo_total_ms": tiempo_total
        }
        
    except Exception as e:
        print(f"🔥 Error en /predict_batch: {e}")
        raise HTTPException(status_code=500, detail=f"Error procesando CSV: {str(e)}")

# ------------------------------------------------------------
# 8. ENDPOINT DE ESTADÍSTICAS (/stats)
# ------------------------------------------------------------
@app.get("/stats", response_model=StatsResponse)
async def obtener_estadisticas():
    """Devuelve estadísticas de uso de la aplicación"""
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            # Total predicciones
            cursor = await db.execute("SELECT COUNT(*) FROM predicciones")
            total = (await cursor.fetchone())[0]
            
            if total == 0:
                return StatsResponse(
                    total_predicciones=0,
                    precio_promedio=0.0,
                    ultima_prediccion=None,
                    distrito_mas_consultado=None
                )
            
            # Precio promedio
            cursor = await db.execute("SELECT AVG(precio_predicho) FROM predicciones")
            precio_prom = (await cursor.fetchone())[0]
            
            # Última predicción
            cursor = await db.execute("SELECT timestamp FROM predicciones ORDER BY id DESC LIMIT 1")
            ultima = (await cursor.fetchone())[0]
            
            # Distrito más consultado
            cursor = await db.execute('''
                SELECT distrito, COUNT(*) as c 
                FROM predicciones 
                GROUP BY distrito 
                ORDER BY c DESC 
                LIMIT 1
            ''')
            row = await cursor.fetchone()
            distrito_pop = row[0] if row else None
            
            return StatsResponse(
                total_predicciones=total,
                precio_promedio=round(precio_prom, 2) if precio_prom else 0.0,
                ultima_prediccion=ultima,
                distrito_mas_consultado=distrito_pop
            )
    except Exception as e:
        print(f"🔥 Error en /stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ------------------------------------------------------------
# 9. ENDPOINT DE SALUD (/health)
# ------------------------------------------------------------
@app.get("/health")
async def health_check():
    """Verifica que el servicio está operativo"""
    return {
        "status": "OK",
        "modelo": "Random Forest (Paris Housing) Pro",
        "db_connected": os.path.exists(DB_PATH)
    }

# ------------------------------------------------------------
# 10. ARRANQUE DEL SERVIDOR
# ------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",  # Importante: 0.0.0.0 para Docker
        port=8000,
        reload=False  # En producción debe ser False
    )