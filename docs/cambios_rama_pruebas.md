# Cambios rama de pruebas — Feature Engineering & Pipeline
**NovaPay ML · Mayo 2026**

---

## Resumen de métricas

| Versión | PR-AUC | Recall | Precision | FP | FN |
|---|---|---|---|---|---|
| Modelo en producción (v1 sobre datos v2) | 0.4701 | 98% | 21% | 1.083 | 6 |
| v2 original (antes de esta rama) | 0.9557 | 94% | ~80% | ~68 | ~18 |
| + Cambios bloque 1 (limpieza FE) | 0.9629 | 94.0% | 80.6% | 68 | 18 |
| **+ Cambios bloque 2 (mejoras FE)** | **0.9620** | **95.7%** | **76.2%** | **90** | **13** |

El avance más importante es pasar de 1.083 FP a 90 y de 18 FN a 13 respecto al modelo v2 de partida.

---

## Bloque 1 — Correcciones y limpieza

### 1.1 Bug crítico: orden Scaler/Imputer en `app.py`

El pipeline de inferencia aplicaba Imputer antes que Scaler, al revés que el pipeline de entrenamiento en `regenerate_models.py`. Esto provocaba que los valores imputados (mediana) se escalaran con la distribución entrenada, introduciendo ruido sistemático en producción.

**Corrección:** intercambiados PASO 3 y PASO 4 en `app.py` para que el orden sea Scale → Impute, igual que en entrenamiento.


### 1.2 Eliminar `high_ratio_redondeado`

Feature que combinaba `txn_vs_limit_pct > 0.85` con `importe % 100 < 1`. La condición del importe redondo destruía la señal del ratio: solo aparecía en el 0.3% de los fraudes. Eliminada del `transform()` de `FeatureEngineer`.

La señal del ratio alto no se pierde — `txn_vs_limit_pct` sigue presente como feature independiente.

### 1.3 `diff_importe_cliente` → z-score real

Implementación anterior:
```python
|importe – media_cliente| / media_cliente   # depende del fit, cold start = 0
```

Nueva implementación usando campos directos del dataset:
```python
(importe – importe_medio_mensual) / desviacion_estandar_mensual
```

Ventajas: no depende del fit (sin cold start), tiene signo (positivo = transacción más grande de lo habitual, señal directa de fraude), y contextualiza correctamente clientes con alta variabilidad frente a clientes con gasto estable.

---

## Bloque 2 — Mejoras de feature engineering

### 2.1 Bug: `night_velocity` (operador de precedencia)

```python
# Antes — .astype(int) solo se aplicaba a (alta_velocidad == 1)
X['night_velocity'] = (X['is_night'] == 1) & (X['alta_velocidad'] == 1).astype(int)

# Después — conversión sobre toda la expresión
X['night_velocity'] = ((X['is_night'] == 1) & (X['alta_velocidad'] == 1)).astype(int)
```

### 2.2 Denominador de `txn_intensity` y `txn_por_minuto`

El `+1` en el denominador (segundos) aplana artificialmente el rango bajo: 0s → `n/1`, 2s → `n/3`. Reemplazado por `clip(lower=30)`, que establece un mínimo de 30 segundos y elimina el artefacto sin distorsionar valores normales.

```python
# Antes
X['txn_intensity']  = n_txns / (tiempo + 1)
X['txn_por_minuto'] = n_txns / (tiempo / 60 + 1)

# Después
X['txn_intensity']  = n_txns / tiempo.clip(lower=30)
X['txn_por_minuto'] = n_txns / (tiempo.clip(lower=30) / 60)
```

### 2.3 `diff_hora_cliente` — distancia circular

La implementación anterior usaba diferencia aritmética: un cliente que opera habitualmente a las 23h y hace una transacción a las 00h obtenía una diferencia de 23 horas cuando la diferencia real es 1 hora.

```python
# Antes
X['diff_hora_cliente'] = (hora - media_hora_cliente).abs()

# Después — distancia circular módulo 24
diff_h = hora - media_hora_cliente
X['diff_hora_cliente'] = diff_h.abs().clip(upper=12).where(diff_h.abs() <= 12, 24 - diff_h.abs())
```

### 2.4 Dos nuevas features de interacción

`cross_border_high_risk`: combinación de operación transfronteriza y cuenta destino de alto riesgo. Cada señal por separado ya tiene poder discriminativo alto; juntas son casi certeza de fraude según el análisis de distribuciones del dataset.

```python
X['cross_border_high_risk'] = (
    (X['cross_border'] == 1) & (X['destino_alto_riesgo'] == 1)
).astype(int)
```

`burst_cross_border`: velocidad alta de transacciones combinada con operación transfronteriza. Captura el patrón de uso masivo de tarjeta desde el extranjero en ventana corta.

```python
X['burst_cross_border'] = (
    (X['burst_rapido'] == 1) & (X['cross_border'] == 1)
).astype(int)
```

El número de features pasa de 67 a 69. LightGBM gana peso en el ensemble (0.50 → 0.67), lo que indica que aprovecha mejor las nuevas interacciones que XGBoost.

---

## Archivos modificados

| Archivo | Cambio |
|---|---|
| `app.py` | Orden Scale→Impute corregido; MODEL_PATH actualizado a v2 |
| `feature_engineering.py` | Eliminado `high_ratio_redondeado`; z-score en `diff_importe_cliente`; fix `night_velocity`; fix denominadores; distancia circular en `diff_hora_cliente`; nuevas features `cross_border_high_risk` y `burst_cross_border` |
| `scripts/feature_engineering.py` | Sincronizado con raíz |
| `model/modelo_08_v2.pkl` | Reentrenado con todos los cambios anteriores |

---

## Pendiente para el siguiente ciclo

- Generar dataset v3 con proporción de fraude realista (~3%) — ver `docs/ampliacion_dataset_v3.md`
- Reentrenar sobre dataset v3 una vez generado
- Implementar `diff_tiempo_cliente` (z-score del tiempo entre transacciones por cliente) — ver sección 4.2 de `docs/analisis_residuos_recomendaciones.md`
- Evaluar Recall@K cuando se defina la cola de revisión manual y el volumen diario de transacciones
