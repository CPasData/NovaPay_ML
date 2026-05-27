# Auditoría de Dependencias — NovaPay ML

Revisión de qué librerías se importan realmente en el proyecto frente a lo que hay declarado en `requirements.txt` y `requirements-doc.txt`.

---

## Resumen del problema

`requirements.txt` tiene más de 200 entradas con duplicados, versiones conflictivas y librerías que no se usan en ningún fichero del proyecto: `torch`, `catboost`, `streamlit`, `Flask`, `playwright`, `selenium`, `statsmodels`, `shap`, `pymongo`, `kagglehub`…

El fichero que realmente importa para producción es `requirements-doc.txt`, que es el que usa el Dockerfile. Ese está mejor, pero también tiene margen de limpieza.

---

## Lo que realmente se usa

### Producción — lo que va en el Dockerfile (`requirements-doc.txt`)

Solo lo que necesita `app.py` y `feature_engineering.py` para arrancar y servir predicciones:

```
fastapi
uvicorn[standard]
pydantic
joblib
pandas
numpy
scikit-learn     # KNNImputer, StandardScaler, BaseEstimator, TransformerMixin
lightgbm
xgboost
```

Nada más. La API es stateless y no conecta a base de datos (guardar_en_bd está comentado), así que `psycopg2` tampoco hace falta por ahora.

### Desarrollo / Scripts de entrenamiento

Lo anterior más:

```
scipy            # ks_2samp en evaluacion_rondas.py
matplotlib
seaborn
```

### Notebooks de análisis

Lo mismo que desarrollo. No se usa nada adicional fuera de lo ya listado.

### Generación de datos sintéticos (opcional)

Solo los scripts de `scripts/synthetic_data/`:

```
faker            # generar_dataset_fraude.py y generar_dataset_fraude_v2.py
sdv              # solo scripts/synthetic_data/sdv/generar_dataset_sdv.py
```

Estos no necesitan estar en el Dockerfile ni en el entorno de producción.

---

## Lo que NO se usa y puede eliminarse

Confirmado buscando imports en todos los `.py` y notebooks del proyecto:

| Librería | Por qué estaba | Se puede eliminar |
|---|---|---|
| `catboost` | Alternativa al ensemble que no se usó | Sí |
| `torch`, `torchaudio`, `torchvision`, `torch-geometric` | Nunca importado | Sí |
| `streamlit` | Posible prototipo de UI | Sí |
| `Flask` | Alternativa a FastAPI descartada | Sí |
| `shap` | Interpretabilidad, no implementada | Sí |
| `statsmodels`, `pmdarima` | Series temporales, no usadas | Sí |
| `imbalanced-learn` / `imblearn` | Resampling, no usado | Sí |
| `psycopg2-binary` | BD comentada en app.py | Sí (por ahora) |
| `pymongo` | No hay MongoDB en el proyecto | Sí |
| `playwright`, `selenium`, `undetected-chromedriver` | Scraping / testing E2E | Sí |
| `kagglehub`, `kagglesdk` | Descarga de datasets | Sí |
| `requests`, `aiohttp`, `beautifulsoup4` | Scraping / HTTP | Sí |
| `plotly`, `altair` | Visualización alternativa no usada | Sí |
| `scikit-image`, `pillow`, `tifffile` | Procesado de imágenes | Sí |
| `networkx` | Grafos, no usados | Sí |
| `phik`, `squarify` | EDA de terceros, no en el código | Sí |
| `numba` | Aceleración numérica, no usada | Sí |
| `openpyxl`, `xlrd`, `pyarrow`, `fastavro` | I/O alternativo, no usado | Sí |
| `GitPython` | No se usa en ningún script | Sí |
| `graphviz` | Visualización de árboles (duplicado además) | Sí |

---

## `requirements-doc.txt` propuesto

Sustituir el contenido actual por:

```
fastapi==0.111.0
uvicorn[standard]==0.29.0
pydantic==2.7.1
joblib==1.4.2
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.5.1
lightgbm==4.6.0
xgboost==2.1.1
```

> Las versiones son las que ya había en el `requirements-doc.txt` original donde coincidían, o las del entorno instalado. Revisar con `pip freeze` en el entorno de entrenamiento para fijarlas exactamente.

## `requirements.txt` propuesto (entorno de desarrollo)

```
# API
fastapi==0.111.0
uvicorn[standard]==0.29.0
pydantic==2.7.1

# ML core
joblib==1.4.2
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.5.1
lightgbm==4.6.0
xgboost==2.1.1

# Análisis y notebooks
scipy
matplotlib
seaborn

# Generación de datos sintéticos (opcional)
faker
sdv
```

---

*Auditoría realizada sobre los ficheros `.py` y `.ipynb` del proyecto, excluyendo `.venv`. La librería `sdv` se detectó solo en `scripts/synthetic_data/sdv/generar_dataset_sdv.py` y puede omitirse si ese script no está en uso activo.*
