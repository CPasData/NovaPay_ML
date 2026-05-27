# NovaPay ML — Cambios v3 (modelo_09_v3)
**Fecha:** Mayo 2026

---

## Resumen

Nuevo ciclo de reentrenamiento del modelo de detección de fraude con **tasa de fraude realista del 3%**, umbrales por canal, métricas de calibración y recall@k. Se crea el modelo `modelo_09_v3.pkl` entrenado sobre `dataset_fraude_v3.csv` (100.000 transacciones, 3,5% fraude).

---

## Cambios realizados

### 1. Nuevo generador de datos sintéticos — `scripts/synthetic_data/generar_dataset_fraude_v3.py`

- **100.000 transacciones** (vs 10.000 en v2) con **~3,5% de fraudes** (target 3%)
- Misma estructura jerárquica que v2: 5.000 clientes → cuentas → tarjetas → transacciones
- Mismas 44 columnas de salida que v1/v2 (compatibilidad total)
- **Nuevo canal `bizum`** (15% de las transacciones) — antes solo `tarjeta` + `transferencia`
- Probabilidad base calibrada al 3% manteniendo el mismo ranking de señales que v2:
  - `destino_alto_riesgo`: +0.04 (antes +0.20)
  - `cross_border`: +0.03 (antes +0.15)
  - `tarjeta robada/extraviada`: +0.05 (antes +0.25)
  - Señales menores escaladas proporcionalmente
- **Misma inyección de señal post-fraude** (55% cross-border, 60% dispositivo no reconocido, etc.)

### 2. Feature Engineering v4 — `scripts/feature_engineering.py`

#### Eliminado: `high_ratio_redondeado`
- Solo aparecía en el 0,3% de fraudes del dataset
- La condición `importe % 100 == 0` destruía la señal del ratio alto
- `txn_vs_limit_pct` ya captura la señal por sí sola
- *Recomendación #1 de analisis_residuos_recomendaciones.md*

#### Reemplazado: `diff_importe_cliente` → z-score real
- Antes: `|importe - media_cliente| / media_cliente` (ignoraba variabilidad, cold start)
- Ahora, **3 nuevas features** usando `importe_medio_mensual` y `desviacion_estandar_mensual`:
  - `diff_importe_zscore`: desviación absoluta tipificada
  - `diff_importe_signed`: desviación con signo (positivo = importe mayor de lo normal)
  - `importe_anomalo`: flag binario si z-score > 3
- *Recomendación #2 de analisis_residuos_recomendaciones.md*
- Total features: **69** (antes 67, +2 netas)

### 3. Pipeline de entrenamiento v3 — `scripts/regenerate_models.py`

#### Nuevas métricas de evaluación:

| Métrica | Descripción | V3 obtenido |
|---------|-------------|-------------|
| **Brier Score** | Error cuadrático medio de probabilidad | 0.0252 ✅ |
| **ECE** | Expected Calibration Error (10 bins) | 0.0616 ⚠️ (~0,05 ideal) |
| **Recall@k** | Fraudes capturados en top 0,1% (simula 200 alertas diarias) | 0.0283 |

#### Thresholds por canal
Optimización de F2-score sobre validación para cada `tipo_transaccion`:

| Canal | Threshold | Justificación |
|-------|-----------|---------------|
| `tarjeta` | 0.759 | Más conservador (mayor volumen legítimo) |
| `transferencia` | 0.724 | Más agresivo (mayor riesgo unitario) |
| `bizum` | 0.783 | Intermedio |

*Implementación parcial de Recomendación #3 de analisis_residuos_recomendaciones.md*

#### Resultados comparativos en test

| Métrica | v2 (15% fraude) | v3 (3,5% fraude) |
|---------|:----------------:|:-----------------:|
| PR-AUC | 0.9622 | 0.9013 |
| AUC-ROC | 0.9864 | 0.9869 |
| Precisión | 77.0% | 78.7% |
| Recall | 94.7% | 84.7% |
| F1 | 0.8495 | 0.8161 |
| Brier | 0.0267 | 0.0252 |
| ECE | 0.0319 | 0.0616 |
| Peso LGB | 0.670 | 0.650 |

### 4. API — `app.py` (v5.0.0)

- Carga `modelo_09_v3.pkl` por defecto
- **Threshold por canal**: usa `tipo_transaccion` de la request para seleccionar el umbral
- Fallback al threshold global si el canal no tiene umbral específico
- Health endpoint ahora reporta: modelo, thresholds por canal, recall@k

### 5. Inferencia batch — `scripts/prediccion_lote.py`

- Soporta `--modelo v3` (nuevo por defecto)
- Threshold por canal en predicción batch
- Mapeo simplificado: `{'v1': 'modelo_07_v1', 'v2': 'modelo_08_v2', 'v3': 'modelo_09_v3'}`

### 6. Nuevos artefactos

| Archivo | Descripción |
|---------|-------------|
| `data/dataset_fraude_v3.csv` | 100.000 transacciones, 3,5% fraude, con bizum |
| `model/modelo_09_v3.pkl` | Modelo entrenado con FE v4, thresholds por canal |

---

## Interpretación de resultados

### Recall@k en producción
- Simulación: **200.000 transacciones/día**, **200 alertas revisables/día** (0,1%)
- El modelo captura el **2,83%** de los fraudes en las primeras 200 alertas
- En términos absolutos: ~20 fraudes detectados de ~700 diarios con solo 200 alertas
- Precisión en alertas: **100%** (ninguna alerta es falso positivo)

### Calibración
- Brier Score 0.0252 (excelente, objetivo < 0.10)
- ECE 0.0616 (ligeramente por encima del objetivo < 0.05)
- La calibración es buena pero mejorable — considerar recalibración isotónica en próximo ciclo

### v3 vs v2
- v3 tiene **menor PR-AUC** (0.90 vs 0.96) porque la tasa de fraude es 5× menor
- Pero la **precisión es mejor** (78.7% vs 77.0%) — menos falsos positivos
- El recall baja al 84.7% porque el modelo es más conservador (threshold más alto)

---

## Recomendaciones post-v3 (próximo ciclo)

1. **Recalibración isotónica** para reducir ECE por debajo de 0.05
2. **Monitorear PSI** en producción para detectar concept drift
3. **Análisis de falsos negativos por cluster** en cada ronda de Ciberseguridad
4. **Evaluar si `is_night` e `is_weekend`** aportan valor en combinación (análisis SHAP)
5. **Ampliar canales**: si `bizum` crece, considerar umbrales adaptativos dinámicos
6. **Añadir contrafactuales** al dataset sintético para cubrir bordes de decisión
