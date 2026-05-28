
# 🛡️ **<font color="#1661a7"> NovaPay ML — Operación Centinela </font>**

**Sistema de detección de fraude en transacciones bancarias desarrollada en tres rondas adversariales.**


---

## **Sobre este repositorio**

Este repositorio pertenece al equipo de **Data Science (Blue Team)** del Desafío de Tripulaciones.

El proyecto **Operación Centinela** es un trabajo conjunto de tres equipos, cada uno con su propio repositorio:

| Equipo | Rol | Repositorio |
|--------|-----|-------------|
| 🔵 **Data Science** (este repo) | Modelo ML + API de detección | `Este repositorio` |
| 🔴 **Ciberseguridad** | Indicar patrones del fraude | `novapay-ciber` |
| 🟢 **Full Stack Back end** | Herramienta del analista (app web) | https://github.com/BV-Works/grupo2-desafio-be |
| 🟢 **Full Stack Front end** | Herramienta del analista (app web) | https://github.com/BV-Works/grupo2-desafio-fe |

> **Integrantes Data Science:** Pablo Da Cunha · Carlos Pascual · Juan Antonio Muñoz · Nadia Llamoca · Alexsandro Luiz

---

## **El Problema**

NovaPay es una fintech ficticia que procesa al rededor de 200.000 transacciones al día. Aproximadamente un 3% (6.000) son fraudulentas, pero el equipo de analistas solo puede revisar 1.000 alertas al día. Sin un sistema automático, la mayoría de fraudes pasan desapercibidos.

**El problema tiene 3 dimensiones:**

| Dimensión | Impacto |
|---|---|
| **Volumen** | 200K tx/día es imposible de revisar manualmente |
| **Velocidad** | Cada transacción debe evaluarse en <100ms para no bloquear la operativa |
| **Precisión** | Alertas falsas saturan a los analistas; fraudes no detectados son pérdida directa |

---

## **¿Qué valor aporta?**

| Indicador | Valor | Qué significa |
|---|---|---|
| **Transacciones pendientes de análisis** | 71.8% | El modelo bloquea sin intervención humana (4.300 fraudes/día) |
| **Fraudes recuperados vía analistas** | 12.9% | Analistas revisan el top k y recuperan 774 fraudes más |
| **Recall total** | 84.7% | De cada 100 fraudes, 85 son detectados |
| **Falsos positivos** | 21.3% | 1 de cada 5 alertas requiere liberación manual |
| **Tiempo de inferencia** | <100 ms | La decisión llega antes de que el cliente termine la operación |
| **ROI estimado** | 3.100:1 | Por cada euro invertido en analistas se recuperan 3.100€ en fraude |

---
## **Arquitectura del Sistema**
![alt text](<Arquitectura proyecto.png>)


---

## **Generación de datos sintéticos**

Los datos no son reales. El equipo de Data Science generó datos sintéticos en **dos etapas**:

**Etapa 1 — Data Science investiga patrones reales:**
El equipo investigó patrones de fraude reales en otros datasets públicos (Kaggle, etc.) e implementó un generador con comportamiento jerárquico: cliente → cuenta → tarjeta → transacción. La primera versión tenía señal débil y 15% de fraude(no realista) PR-AUC 0.35.

**Etapa 2 — Ciberseguridad aporta nuevos patrones:**
El Red Team revisó el dataset y aportó patrones de fraude más realistas: ráfagas de transacciones, dispositivos no reconocidos, operaciones transfronterizas coordinadas, importes cerca del límite. Con estos patrones Data Science generó la versión mejorada (PR-AUC 0.96).

**Etapa 3.5 — Se incluyen nuevos patrones más sofisticados:**
Se incorporaron nuevas estructuras de fraude: como el patron burst, donde se realizan varias transacciones en un corto periódo de tiempo, también se introdujeron patrones donde los importes eran menores, así como una estructura más sigilosa, donde no se activaban demasiadas flags.
(PR-AUC 0.90).

```
Data Science investiga patrones de fraude reales
        ↓
Genera dataset_fraude_v1.csv (10K tx, señal débil)
        ↓
Ciberseguridad revisa y aporta nuevos patrones
        ↓
Data Science incorpora los patrones de Ciber
→ Inyección de señal post-hoc
→ Distribuciones genuinamente distintas fraude vs legítimo
        ↓
dataset_fraude_v2.csv (10K tx) → PR-AUC: 0.35 → 0.96
dataset_fraude_v3.csv (100K tx, 3% fraude realista)
```

---

## **El Modelo ML**

### Iteraciones del modelo

El equipo realizó **3 iteraciones** hasta alcanzar el rendimiento actual:

| Iteración | Dataset | Tasa fraude | PR-AUC | Mejora aplicada |
|-----------|---------|-------------|--------|-----------------|
| **Modelo 1** | 10K tx | 15% | 0.35 | Línea base — datos sintéticos simples |
| **Modelo 2** | 10K tx | 15% | 0.96 | Inyección de señal post-hoc con patrones de Ciber |
| **Modelo 3** | 100K tx | 3.5% | 0.90 | Tasa realista + threshold por canal de pago |

> El salto más importante fue del Modelo 1 al 2: de PR-AUC 0.35 a 0.96 gracias a generar datos donde las features de fraude tienen distribuciones genuinamente diferentes a las legítimas.

### **Arquitectura del modelo**

```
POST /predict
     ↓
┌─ Feature Engineering ──────────────┐
│  44 columnas crudas → 69 features  │
└──────────┬─────────────────────────┘
           ↓
┌─ StandardScaler ───────────────────┐
│  Normaliza features numéricas      │
└──────────┬─────────────────────────┘
           ↓
┌─ KNNImputer (n=5) ─────────────────┐
│  Rellena nulos con vecinos         │
└──────────┬─────────────────────────┘
           ↓
┌─ Ensemble ─────────────────────────┐
│  LightGBM (65%) + XGBoost (35%)    │
└──────────┬─────────────────────────┘
           ↓
┌─ Threshold por canal ──────────────┐
│  tarjeta:       0.759              │
│  transferencia: 0.724              │
│  bizum:         0.783              │
└──────────┬─────────────────────────┘
           ↓
Respuesta con 8 campos de predicción
```
### **Datos: las 69 features (v4)**

Las features transformadas se organizan en 4 bloques:

| Bloque | Features | Ejemplo |
|---|---|---|
| **Ratios financieros** (8) | `txn_vs_limit_pct`, `outflow_inflow_ratio`, `net_flow_30d`, `balance_utilization`, `saldo_ratio_ingreso`, `saldo_ratio_limite`, `txn_ratio_media` | % del límite usado en una transacción |
| **Flags geográficos** (4) | `cross_border`, `foreign_unknown_device`, `foreign_known_device`, `domestic_unknown_device` | Cruce país + dispositivo → 43.6% de fraudes |
| **Features de sesión** (7) | `txn_por_minuto`, `burst_rapido`, `alta_velocidad`, `txn_intensity`, `txn_severity`, `actividad_alta`, `ratio_dispositivo_hora` | Ráfagas de >5 tx en <5 min |
| **Desviación temporal** (7) | `diff_importe_zscore`, `diff_importe_signed`, `importe_anomalo`, `dias_desde_ultimo_pin`, `dias_desde_ultimo_cambio`, `antiguedad_relativa_media`, `diff_antiguedad_tx` | z-score real con media y desviación del cliente |

Las 69 features restantes son codificaciones target, frecuencias, flags categóricos y combinaciones de sesión.

### **Modelo: arquitectura interna**

Cada `.pkl` contiene un pipeline completo:

```
modelo.pkl = {
    'fe':      FeatureEngineer      # sklearn Transformer → 44 → 69 features
    'scaler':  StandardScaler       # Normaliza features numéricas
    'imputer': KNNImputer (n=5)     # Imputa nulos por similitud
    'lgb_model': LightGBM           # Boosting rápido, captura outliers
    'xgb_model': XGBoost           # Boosting robusto, maneja desbalance
    'best_w':   float               # Peso del ensemble (ej: 0.65 = 65% LGB)
    'best_t':   float               # Threshold global (ej: 0.7622)
    'num_feats': list[str]          # Features numéricas para scaler/imputer
    'per_channel_thresholds': dict  # Threshold por canal (tarjeta/transferencia/bizum)
    'metadata': dict                # Métricas adicionales (recall@k, etc.)
}
```


### **Métricas del modelo**

| Métrica | Valor | Target | Interpretación |
|---------|-------|--------|----------------|
| **PR-AUC** | 0.90 | >0.85 | Principal métrica de ranking |
| **AUC-ROC** | 0.987 | >0.97 | Separación fraude/legítimo |
| **F2-score** | 0.67 | >0.60 | Prioriza recall sobre precisión |
| **Recall global** | 84.7% | >80% | % de fraudes detectados |
| **Recall@k (k=0.5%)** | 12.9% | >10% | % fraudes capturados en 1.000 alertas revisables |
| **Precisión** | 78.7% | >75% | % de alertas que son fraude real |
| **Brier Score** | 0.033 | <0.05 | Calibración de probabilidades |

---

## **API FastAPI**

### **Endpoints**

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| `GET` | `/health` | Estado de la API y modelo activo |
| `POST` | `/predict` | Predice UNA transacción |
| `POST` | `/predict/batch` | Predice MUCHAS transacciones a la vez |

### **Respuesta de `/predict`**

```json
{
  "id_transaccion"    : "tx-001",
  "is_fraud"          : 1,
  "prob_fraud"        : 0.8734,
  "impacto_fraude"    : 3,
  "es_transfronteriza": 1,
  "ratio_imp_limite"  : 0.95,
  "intensidad_tx"     : 0.50,
  "severidad_tx"      : 4500.0,
  "flujo_neto_30d"    : -4500.0,
  "mensaje"           : "FRAUDE DETECTADO - probabilidad fraude 87%"
}
```

---

## **Stack tecnológico**

| Capa | Tecnología | Versión |
|------|-----------|-----------|
| **Lenguaje** | Python | 3.11+ |
| **API** | FastAPI + Uvicorn | 0.136 + 0.047 |
| **Validación** | Pydantic v2 |2.13 |
| **ML Core** | scikit-learn |1.8 |
| **Gradient Boosting** | LightGBM + XGBoost |— |
| **Serialización** | joblib |1.5 |
| **Datos** | pandas, numpy, scipy | 3.0, 2.4, 1.17 |
| **Container** | Docker + docker-compose |— |
| **Despliegue** | AWS EC2 |— |
| **Base de datos** | PostgreSQL (gestionada por Full Stack) |

---

## **Estructura del proyecto**

```
C:\Dev\NovaPay_ML\
│
├── app.py                               # FastAPI
├── dockerfile                           
├── docker-compose.yml                   
├── requirements.txt                     # Dependencias Python
├── requirements-doc.txt                 # Dependencias para docker
│
├── scripts/                    # Código principal
│   ├── feature_engineering.py           
│   ├── regenerate_models.py             # Entrenamiento completo v1+v2+v3
│   ├── prediccion_lote.py               # Inferencia batch CSV → CSV
│   ├── inference_example.py             # Ejemplo de carga + inferencia
│   ├── evaluacion_rondas.py             # Simulación de producción con drift
│   ├── generar_muestra_sin_etiqueta.py  # Datos sin target (3 perfiles)
│   ├── generar_lote_prediccion.py       # Combina datasets para inferencia
│   ├── separar_prediccion_json.py       # CSV → JSON separado
│   ├── aplanar_predicciones.py          # Aplana a 10 campos
│   │
│   └── synthetic_data/         
│       ├── generar_dataset_fraude.py      # v1
│       ├── generar_dataset_fraude_v2.py   # v2 
│       └── generar_dataset_fraude_v3.py   # v3
│
├── model/                      # Modelos entrenados (.pkl autocontenidos)
│   ├── modelo_07_v1.pkl               # v1 — 15% fraude, señal débil
│   ├── modelo_08_v2.pkl               # v2 — 15% fraude, señal mejorada
│   └── modelo_09_v3.pkl               # v3 — 3% fraude, thresholds por canal
│
├── data/                       # Datasets
│   ├── dataset_fraude.csv            # v1 (10K tx)
│   ├── dataset_fraude_v2.csv         # v2 (10K tx)
│   ├── dataset_fraude_v3.csv         # v3 (100K tx, dataset principal)
│   └── muestra_sin_etiqueta.json     # Muestras sin etiqueta para pruebas
│
├── notebooks/                  # Jupyter notebooks (entrenamiento + EDA)
│   └── 07_train_fraud_v1.ipynb
│   └── 08_train_fraud_v2.ipynb
│   └── EDA_v2.ipynb
│   └── EDA_v3.ipynb
│   └── 09_pipeline_completo_v1.ipynb
│   └── 09_pipeline_completo.ipynb  # Pipeline completo v3
│
└── docs/                       # Documentación
    ├── guion_presentacion.md        # Guión completo de presentación
    ├── comandos.md                  # Todos los comandos
    ├── 03_feature_engineering_deep_dive.ipynb
    ├── 06_model_selection_deep_dive.ipynb
    └── residuos_y_evaluacion.md
    └── NOVAPAY.pptx
    └── test_api.ipynb               # Notebook para pruebas de API
```

---

## **Instalación y uso**

### **Local sin Docker**

```powershell
# Entorno virtual con uv
uv venv
.venv\Scripts\activate
uv pip install -r requirements.txt

# Arrancar la API
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### **Con Docker**

```powershell
docker-compose up --build
```

La API estará en `http://localhost:8000` | Swagger en `http://localhost:8000/docs`

---

## **Criterios de éxito**

- Pipeline end-to-end funcionando
- Mejora medible entre iteraciones del modelo
- Coordinación entre equipos con interfaces acordadas
- Herramienta del analista instructiva y clara
- Documentación técnica completa

---

## **Licencia y Derechos de Autor**

Este proyecto ha sido desarrollado con fines educativos por estudiantes.

Se distribuye bajo la licencia MIT, lo que permite su uso, copia, modificación y distribución, siempre que se incluya la atribución correspondiente a los autores.

© 2026 – Pablo Da Cunha · Carlos Pascual · Juan Antonio Muñoz · Nadia Llamoca · Alexsandro Luiz

---

*Desafío de Tripulaciones — TheBridge | Mayo 2026*
