# 🏠 Paris Housing Price Predictor

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-red.svg)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-✓-2496ED.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Sistema de predicción de precios inmobiliarios en París usando Machine Learning (Random Forest) con API REST, interfaz web y procesamiento batch.**

---

## 📋 Tabla de Contenidos

- [Características](#-características)
- [Stack Tecnológico](#-stack-tecnológico)
- [Instalación Rápida](#-instalación-rápida)
- [Uso](#-uso)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [API Endpoints](#-api-endpoints)
- [Entrenamiento del Modelo](#-entrenamiento-del-modelo)
- [Despliegue con Docker](#-despliegue-con-docker)
- [Contribuciones](#-contribuciones)
- [Licencia](#-licencia)

---

## ✨ Características

- **🧠 Modelo de Machine Learning:** Random Forest Regressor entrenado con 1,200 propiedades reales de París.
- **⚡ API REST:** Backend en FastAPI con endpoints para predicciones individuales y batch.
- **🎨 Interfaz Web:** Frontend en Streamlit con formulario interactivo y subida de archivos CSV.
- **💾 Persistencia:** Base de datos SQLite para almacenar historial de predicciones.
- **📊 Estadísticas en tiempo real:** Métricas de uso, precio promedio y distritos más consultados.
- **🐳 Dockerizado:** Despliegue con un solo comando usando Docker Compose.
- **📁 Procesamiento Batch:** Sube un CSV con cientos de propiedades y descarga los resultados.

---

## 🛠 Stack Tecnológico

| Capa | Tecnología | Propósito |
|------|------------|-----------|
| **Data Science** | Pandas, NumPy, Scikit-learn | Análisis y entrenamiento del modelo |
| **Backend** | FastAPI, Uvicorn, Pydantic | API REST y lógica de negocio |
| **Frontend** | Streamlit, Requests | Interfaz de usuario web |
| **Base de Datos** | SQLite, aiosqlite | Persistencia de predicciones |
| **DevOps** | Docker, Docker Compose | Contenerización y despliegue |
| **Desarrollo** | Jupyter Notebook | Experimentación y entrenamiento |

---

## 🚀 Instalación Rápida

### Requisitos Previos

- [Docker](https://www.docker.com/products/docker-desktop/) y Docker Compose
- (Opcional) Python 3.12+ para desarrollo local

### Método 1: Con Docker (Recomendado)

```
# 1. Clonar el repositorio
git clone https://github.com/tu-usuario/paris-housing-predictor.git
cd paris-housing-predictor

# 2. (Opcional) Configurar variables de entorno
cp .env.example .env

# 3. Levantar los servicios
docker-compose up -d

# 4. Acceder a la aplicación
# Frontend: http://localhost:8501
# API Docs: http://localhost:8000/docs
```
### Método 2: Desarrollo Local (Sin Docker)
```
# 1. Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Entrenar el modelo (opcional, ya viene entrenado)
cd notebooks
jupyter notebook 01_entrenamiento.ipynb

# 4. Iniciar Backend (Terminal 1)
cd backend
python main.py

# 5. Iniciar Frontend (Terminal 2)
cd frontend
streamlit run app.py
```
#### 📖 Uso

🏢 Predicción Individual
```
1.  Accede a http://localhost:8501

2.  Completa el formulario con las características de la propiedad:

    Distrito (1-20)

    Metros cuadrados

    Número de habitaciones

    Planta

    Año de construcción

    Distancia al centro

    Tipo de propiedad

    Estado de conservación

3.  Haz clic en "Valorar propiedad"

4.  Obtén el precio estimado con rango de confianza
```
📁 Procesamiento Batch (CSV)
```
1.  Prepara un archivo CSV con las siguientes columnas:
Arrondissement,Size_sqm,Rooms,Floor,Year_Built,Distance_to_Center_km,Property_Type,Condition
2.  Ve a la pestaña "Procesamiento Batch"

3.  Sube tu archivo CSV

4.  Descarga el archivo con la columna Precio_Predicho añadida

Ejemplo de CSV válido:
Arrondissement,Size_sqm,Rooms,Floor,Year_Built,Distance_to_Center_km,Property_Type,Condition
8,120,4,3,1995,2.5,Apartment,Good
18,30,1,2,1980,5.0,Studio,Needs Renovation
16,180,5,8,2010,4.0,Penthouse,New
```
📁 Estructura del Proyecto
```
.
├── backend/                    # API FastAPI
│   ├── main.py                # Código principal del servidor
│   └── Dockerfile             # Imagen Docker del backend
│
├── frontend/                   # Interfaz Streamlit
│   ├── app.py                 # Código de la UI
│   └── Dockerfile             # Imagen Docker del frontend
│
├── notebooks/                  # Experimentación y entrenamiento
│   └── 01_entrenamiento.ipynb # Notebook de análisis y modelo
│
├── modelos_entrenados/         # Modelos serializados (.pkl)
│   ├── modelo_precios_paris.pkl
│   ├── mapeo_condition.pkl
│   └── columnas_input.pkl
│
├── data/                       # Datos de ejemplo
│   └── paris_housing_prices_dataset.csv
│
├── docker-compose.yml          # Orquestación de servicios
├── requirements.txt            # Dependencias Python
├── .env.example               # Template de variables de entorno
├── .gitignore                 # Archivos excluidos de Git
└── README.md                  # Este archivo
```
🔌 API Endpoints
```
POST /predict
Predice el precio de una única propiedad.

Request Body:

{
  "Arrondissement": 8,
  "Size_sqm": 120,
  "Rooms": 4,
  "Floor": 3,
  "Year_Built": 1995,
  "Distance_to_Center_km": 2.5,
  "Property_Type": "Apartment",
  "Condition": "Good"
}

Response:

{
  "precio_estimado": 1878944.85,
  "moneda": "EUR",
  "rango_inferior": 1465576.98,
  "rango_superior": 2292312.72
}

POST /predict_batch
Procesa un archivo CSV con múltiples propiedades.

    Content-Type: multipart/form-data

    File field: file

GET /stats
Devuelve estadísticas de uso.

Response:
{
  "total_predicciones": 42,
  "precio_promedio": 1521459.0,
  "ultima_prediccion": "2026-04-22T15:30:00",
  "distrito_mas_consultado": 8
}

GET /health
Health check para monitoreo.

Response:
{
  "status": "OK",
  "modelo": "Random Forest (Paris Housing) Pro",
  "db_connected": true
}
```
🧪 Entrenamiento del Modelo
```
El modelo fue entrenado con el dataset Paris Housing Prices (1,200 registros) de Kaggle.

Métricas de rendimiento:

    R² Score: 0.766 (76.6%)

    MAE (Error Absoluto Medio): 332,524 €

    Features utilizadas: 10 características después de encoding

Preprocesamiento aplicado:

    Eliminación de columnas identificadoras (Property_ID)

    One-Hot Encoding para Property_Type

    Mapeo ordinal para Condition (New=4, Good=3, Renovated=2, Needs Renovation=1)

Para reentrenar el modelo con nuevos datos:

cd notebooks
jupyter notebook 01_entrenamiento.ipynb
# Ejecutar todas las celdas y verificar que se generan los .pkl en modelos_entrenados/
```
🐳 Despliegue con Docker
```
Construir y levantar
    docker-compose up --build -d

Ver logs
    docker-compose logs -f backend
    docker-compose logs -f frontend

Detener servicios (conservando datos)
    docker-compose down

Detener y eliminar todo (incluyendo base de datos)
    docker-compose down -v

Verificar volúmenes
    docker volume ls | grep paris_housing
```
🤝 Contribuciones
```
¡Las contribuciones son bienvenidas! Para contribuir:

1.  Haz un Fork del repositorio
2.  Crea una rama para tu feature (git checkout -b feature/amazing-feature)
3.  Haz commit de tus cambios (git commit -m 'Add amazing feature')
4.  Push a la rama (git push origin feature/amazing-feature)
5.  Abre un Pull Request

Áreas de mejora sugeridas:

    Añadir más features al modelo (ej. orientación, vistas, ascensor)
    Implementar autenticación JWT para la API
    Migrar a PostgreSQL para producción
    Añadir tests unitarios con pytest
    Desplegar en la nube (Render, Railway, AWS)
```
📄 Licencia
```
Este proyecto está bajo la Licencia MIT. Ver archivo [LICENSE](LICENSE) para más detalles.
```
⭐ Agradecimientos
```
    Dataset: Paris Housing Prices
    Inspiración: Proyecto educativo de Machine Learning aplicado

¿Te ha sido útil este proyecto? ¡Dale una estrella ⭐ en GitHub!
