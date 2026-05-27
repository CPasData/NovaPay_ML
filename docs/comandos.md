# NovaPay ML — Todos los comandos

Todos los comandos se ejecutan desde la raíz del proyecto (`C:\Dev\NovaPay_ML`).

---

## 1. Generación de datos sintéticos

### v3 — 100.000 tx, 3% fraude, incluye bizum (dataset principal)

```powershell
python scripts/synthetic_data/generar_dataset_fraude_v3.py
```
Genera `data/dataset_fraude_v3.csv` (100.000 transacciones, ~3.5% fraude).

### v2 — 10.000 tx, ~15% fraude, señal mejorada

```powershell
python scripts/synthetic_data/generar_dataset_fraude_v2.py
```
Genera `data/dataset_fraude_v2.csv` (10.000 transacciones, ~15% fraude).

### v1 — 10.000 tx, ~3% fraude, señal débil (original)

```powershell
python scripts/synthetic_data/generar_dataset_fraude.py
```
Genera `data/dataset_fraude.csv` (10.000 transacciones, ~3% fraude).

### SDV — datos sintéticos con GaussianCopula

```powershell
python scripts/synthetic_data/sdv/generar_dataset_sdv.py
```
Genera `data/dataset_sdv.csv` usando SDV.

---

## 2. Generación de muestras sin etiqueta (para testear API/inferencia)

Genera archivos CSV + JSON sin la columna `IS_FRAUD`.

```powershell
# Perfil mixto (estándar) — 200 transacciones
python scripts/generar_muestra_sin_etiqueta.py
# → data/muestra_sin_etiqueta.csv + .json

# Perfil sospechoso (señales de riesgo elevadas)
python scripts/generar_muestra_sin_etiqueta.py --perfil sospechoso
# → data/muestra_sospechoso.csv + .json

# Perfil fraudulento (señales muy marcadas)
python scripts/generar_muestra_sin_etiqueta.py --perfil fraude
# → data/muestra_fraude.csv + .json

# Los 3 perfiles a la vez
python scripts/generar_muestra_sin_etiqueta.py --perfil todo
# → muestra_mixto, muestra_sospechoso, muestra_fraude (CSV+JSON cada uno)

# Personalizar cantidad y semilla
python scripts/generar_muestra_sin_etiqueta.py --n 500 --seed 123
```

| Argumento | Default | Opciones | Descripción |
|---|---|---|---|
| `--n` | 200 | — | Número de transacciones |
| `--seed` | 42 | — | Semilla aleatoria |
| `--perfil` | `mixto` | `mixto`, `sospechoso`, `fraude`, `todo` | Perfil de riesgo |
| `--output` | `data/muestra_sin_etiqueta` | — | Ruta base (sin extensión) |

---

## 3. Entrenamiento de modelos

Regenera los 3 modelos (v1, v2, v3) completos con Feature Engineering, escalado,
imputación, ensemble LGB+XGB, threshold F2 y métricas.

```powershell
python scripts/regenerate_models.py
```

Genera en `model/` los archivos:
- `modelo_07_v1.pkl` — v1 (original, señal débil)
- `modelo_08_v2.pkl` — v2 (señal mejorada)
- `modelo_09_v3.pkl` — v3 (3% fraude, thresholds por canal, recall@k)

---

## 4. Inferencia batch (CSV → CSV con predicciones)

Lee un CSV de transacciones y genera otro CSV con los 10 campos de predicción.

```powershell
# Usar modelo v3 (default)
python scripts/prediccion_lote.py --input data/muestra_sin_etiqueta.csv --output data/resultados.csv

# Usar modelo v2
python scripts/prediccion_lote.py --input data/dataset_fraude_v2.csv --output data/predicciones_v2.csv --modelo v2

# Usar modelo v1
python scripts/prediccion_lote.py --input data/dataset_fraude.csv --output data/predicciones_v1.csv --modelo v1
```

| Argumento | Requerido | Default | Descripción |
|---|---|---|---|
| `--input` | Sí | — | CSV de entrada con transacciones |
| `--output` | Sí | — | CSV de salida con predicciones |
| `--modelo` | No | `v3` | `v1`, `v2` o `v3` |

---

## 5. API REST (FastAPI)

### Arrancar la API

```powershell
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### Endpoints

#### Health check

```powershell
curl http://localhost:8000/health
```

Respuesta:
```json
{
  "status": "ok",
  "version": "5.0.0",
  "modelo": "modelo_09_v3.pkl",
  "ensemble": "LGB 65% + XGB 35%",
  "threshold_global": 0.7622,
  "thresholds_canal": {"tarjeta": 0.759, "transferencia": 0.724, "bizum": 0.783},
  "recall_at_k": 0.0283
}
```

#### Predecir una transacción

```powershell
curl -X POST http://localhost:8000/predict `
  -H "Content-Type: application/json" `
  -d '{"id_transaccion": "test-01", "id_cliente": "c1", ...}'
```

#### Predecir lote (vía JSON)

```powershell
curl -X POST http://localhost:8000/predict/batch `
  -H "Content-Type: application/json" `
  -d (Get-Content data/muestra_sin_etiqueta.json -Raw)
```

#### Interfaz Swagger (documentación interactiva)

Abrir en el navegador: http://localhost:8000/docs

---

## 6. Evaluación por rondas (simulación de producción)

Genera transacciones en rondas de 100, ejecuta inferencia, acumula métricas
por ronda y detecta drift.

```powershell
# Baseline (sin drift) — 50 rondas
python scripts/evaluacion_rondas.py

# Drift suave (cambio gradual en ronda 20-40)
python scripts/evaluacion_rondas.py --modelo v3 --rondas 50 --drift suave

# Drift abrupto (cambio brusco en ronda 30)
python scripts/evaluacion_rondas.py --modelo v3 --drift abrupto

# Drift de concepto (la relación features→fraude cambia)
python scripts/evaluacion_rondas.py --modelo v3 --drift concepto

# Guardar métricas a CSV
python scripts/evaluacion_rondas.py --output data/metricas_rondas.csv
```

| Argumento | Default | Opciones | Descripción |
|---|---|---|---|
| `--modelo` | `v3` | `v1`, `v2`, `v3` | Modelo a evaluar |
| `--rondas` | 50 | — | Número de rondas |
| `--drift` | `baseline` | `baseline`, `suave`, `abrupto`, `concepto` | Escenario de drift |
| `--output` | `""` | — | CSV de salida con métricas (vacío = solo consola) |
| `--seed` | 42 | — | Semilla aleatoria |

---

## 7. Preparación de lotes para inferencia

Combina datasets existentes y elimina columnas target.

```powershell
# Default: combina dataset_fraude.csv + dataset_fraude_v2.csv
python scripts/generar_lote_prediccion.py

# Personalizar salida
python scripts/generar_lote_prediccion.py --output data/mi_lote.csv
```

| Argumento | Default | Descripción |
|---|---|---|
| `--output` | `data/lote_sin_target.csv` | Ruta del CSV de salida |

---

## 8. Separación de predicciones en JSON

Toma el CSV de `prediccion_lote.py` y lo separa en dos JSON:
`transacciones.json` (datos crudos) y `predicciones.json` (features + predicción).

```powershell
python scripts/separar_prediccion_json.py --input data/resultados.csv
python scripts/separar_prediccion_json.py --input data/predicciones_v2.csv --output-dir data/
```

| Argumento | Requerido | Default | Descripción |
|---|---|---|---|
| `--input` | Sí | — | CSV generado por `prediccion_lote.py` |
| `--output-dir` | No | `data` | Directorio de salida |

---

## 9. Aplanar predicciones a formato compacto

Toma el `predicciones.json` de `separar_prediccion_json.py` y lo aplana
a solo 10 campos clave.

```powershell
python scripts/aplanar_predicciones.py --input data/predicciones.json
python scripts/aplanar_predicciones.py --input data/predicciones.json --output data/aplanado.json
```

| Argumento | Requerido | Default | Descripción |
|---|---|---|---|
| `--input` | Sí | — | `predicciones.json` de `separar_prediccion_json.py` |
| `--output` | No | `data/predicciones_aplanadas.json` | JSON de salida |

---

## 10. Docker

### Construir y ejecutar

```powershell
# Construir imagen
docker build -t novapay-ml:v3 -f dockerfile .

# Ejecutar contenedor
docker run -d --name novapay-api -p 8000:8000 --restart unless-stopped novapay-ml:v3

# Ver logs
docker logs novapay-api

# Parar y eliminar
docker stop novapay-api
docker rm novapay-api
```

### Con docker-compose

```powershell
docker compose up -d
docker compose logs -f
docker compose down
```

---

## 11. Ejemplo de inferencia (script independiente)

```powershell
python scripts/inference_example.py
```
Carga el modelo v3 desde `model/modelo_09_v3.pkl`, ejecuta inferencia sobre
`data/dataset_fraude_v2.csv` y muestra un classification report.

---

## 12. Flujo completo típico

```powershell
# 1. Generar datos de entrenamiento v3
python scripts/synthetic_data/generar_dataset_fraude_v3.py

# 2. Entrenar modelo
python scripts/regenerate_models.py

# 3. Generar muestra sin etiqueta para pruebas
python scripts/generar_muestra_sin_etiqueta.py --perfil todo

# 4. Inferencia batch (CSV → CSV con predicciones)
python scripts/prediccion_lote.py --input data/muestra_sin_etiqueta.csv --output data/resultados.csv

# 5. O arrancar API y probar
uvicorn app:app --host 0.0.0.0 --port 8000
# En otra terminal:
curl -X POST http://localhost:8000/predict/batch -H "Content-Type: application/json" -d (Get-Content data/muestra_sin_etiqueta.json -Raw)

# 6. Evaluar con simulación de producción
python scripts/evaluacion_rondas.py --modelo v3 --rondas 20 --drift suave --output data/metricas.csv

# 7. Empaquetar en Docker
docker build -t novapay-ml:v3 -f dockerfile .
docker run -d -p 8000:8000 novapay-ml:v3
```
