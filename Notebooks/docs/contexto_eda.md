# Contexto del EDA — NovaPay Operación Centinela (Ronda 1)

> Documento de traspaso. Última actualización: **23/05/2026**.
> Autor del análisis: Juan Antonio (Data Science — Blue Team, Grupo 2).
> Este documento está pensado para que un compañero pueda retomar el trabajo sin perder contexto.

> **Estado actual:** el EDA está **completamente cerrado**. El notebook `EDA_ordenado.ipynb` tiene el análisis completo, las interpretaciones, las conclusiones y las recomendaciones de feature engineering y modelado (secciones 9 y 10). **El siguiente paso es crear el notebook de feature engineering** — lo que hay que hacer está detallado en la sección 6.

---

## 1. El dataset

**Archivo:** `Notebooks/data/dataset_fraude.csv`
**Filas:** 10.000 transacciones
**Columnas originales:** 45 (se elimina `IMPACTO_FRAUDE` al inicio del notebook)
**Columnas tras feature engineering temporal:** 48

No hay ningún valor nulo en ninguna columna — el dataset sintético está completo.

### Variable objetivo

`IS_FRAUD` — binaria (0 / 1)
- Tasa de fraude global: **15,26%** (~1.526 transacciones fraudulentas de 10.000)
- El dataset está **desbalanceado** (~5,6:1 legítimas/fraudulentas)

### Cobertura temporal

El dataset cubre solo **enero–mayo y noviembre–diciembre**. Los meses junio–octubre no existen. No hay estacionalidad relevante entre los meses presentes (fraud rate estable entre 12–16%).

### Estructura de columnas por bloque temático

| Bloque | Columnas principales |
|---|---|
| Cliente | `id_cliente`, `tipo_cliente`, `edad_cliente`, `customer_country`, `customer_region`, `tenure`, `importe_medio_mensual`, `desviacion_estandar_mensual`, `media_transacciones_al_dia`, `numero_fraudes_ultimo_ano` |
| Cuenta | `id_cuenta`, `cuenta_origen`, `estado_cuenta`, `saldo_actual`, `saldo_medio_30_dias`, `volumen_entrante_30_dias`, `volumen_saliente_30_dias`, `numero_transferencias_recibidas_7_dias`, `numero_transferencias_enviadas_7_dias` |
| Tarjeta | `id_tarjeta`, `estado_tarjeta`, `fecha_creacion_tarjeta`, `antiguedad_tarjeta_dias`, `limite_importe_transacciones`, `veces_superar_limite_7_dias` |
| Transacción | `id_transaccion`, `tipo_transaccion`, `fecha_hora`, `is_night`, `is_weekend`, `tiempo_desde_ultima_transaccion`, `numero_transacciones_ultima_hora`, `importe_transaccion`, `metodo_autenticacion`, `numero_pin_disponibles` |
| Dispositivo / geolocalización | `identificador_dispositivo_fingerprint`, `dispositivo_reconocido`, `operacion_pais`, `operacion_region`, `direccion_ip_origen`, `geolocalizacion` |
| Destino | `cuenta_destino`, `destino_alto_riesgo` |
| FE temporal (derivadas) | `fecha`, `hora`, `dia_semana`, `mes` |

---

## 2. Lo que está hecho en el notebook

### 2.1 Setup y carga

- Importación de librerías (`numpy`, `pandas`, `matplotlib`, `seaborn`)
- Carga del CSV y visualización con `.head()`
- Eliminación de la columna `IMPACTO_FRAUDE`
- Feature engineering temporal: extracción de `fecha`, `hora`, `dia_semana`, `mes` a partir de `fecha_hora`

### 2.2 Inspección inicial

- `.isna().sum()` → confirmación de 0 nulos en todas las columnas
- `.info()` → tipos de datos revisados (16 int, 8 float, 1 datetime, 20+ str/object, 2 int32 de FE)
- `.describe()` → estadísticos básicos de todas las numéricas
- `.describe()` específico de `importe_transaccion` (media 473€, mediana 148€, max 27.088€ — muy sesgada a la derecha)
- Listado de columnas categóricas identificadas

### 2.3 Análisis temporal de fraude

- **Fraud rate por hora:** bastante plano (~14-16% la mayoría de horas), con dos picos claros en **hora 2** y **hora 12** (~20%). El patrón nocturno esperado NO es tan pronunciado.
- **Fraud rate por día de la semana:** muy homogéneo (≈14-16%), sin un día claramente peor.
- **Fraud rate por mes:** estable entre 12–16% en los meses presentes. Sin estacionalidad relevante.
- **Distribución fraude/no-fraude** en barras para: `tipo_cliente`, `customer_country`, `customer_region`, `estado_cuenta`, `estado_tarjeta`, `tipo_transaccion`, `metodo_autenticacion`, `operacion_pais`, `operacion_region`, `dia_semana`, `hora`, `mes`
- **Heatmap hora × día de la semana:** fraud rate sin patrón claro
- **Heatmap hora × importe_transaccion (tramos):** a mayor importe, más fraude — especialmente en transacciones grandes fuera de horario nocturno
- **Heatmap hora × dispositivo_reconocido:** dispositivos no reconocidos tienen fraud rate notablemente más alto (~21-30%) que los reconocidos (~12-20%)
- **Heatmap hora × destino_alto_riesgo:** destinos de alto riesgo multiplican el fraude en cualquier hora
- **Línea de fraud rate por hora con franja nocturna resaltada:** visualmente el tramo nocturno no destaca de forma especial

### 2.4 Histogramas comparados (fraude vs no fraude)

Variables analizadas con histogramas superpuestos:
- `edad_cliente` — distribución similar entre clases
- `media_transacciones_al_dia` (redondeada)
- `numero_transferencias_enviadas_7_dias` (redondeada)
- `numero_transferencias_recibidas_7_dias` (redondeada)
- `veces_superar_limite_7_dias` (redondeada)
- `numero_transacciones_ultima_hora`

También se calculó la **proporción de fraude por valor discreto** para estas mismas variables (barplots con mínimo 30 observaciones por grupo).

### 2.5 Correlaciones con IS_FRAUD (variables numéricas — Pearson)

Resultados ordenados por valor absoluto:

| Variable | Correlación |
|---|---|
| `destino_alto_riesgo` | **+0.188** |
| `dispositivo_reconocido` | **-0.087** |
| `importe_transaccion` | **+0.084** |
| `numero_fraudes_ultimo_ano` | +0.039 |
| `importe_medio_mensual` | +0.033 |
| resto de variables | < 0.02 (muy débil) |

- Barplot de correlaciones con `IS_FRAUD`
- Heatmap de matriz de correlaciones completa entre todas las numéricas

**Conclusión:** Las correlaciones lineales son débiles en general. Las tres variables más relevantes por Pearson son `destino_alto_riesgo`, `dispositivo_reconocido` e `importe_transaccion`.

### 2.6 Relación de variables categóricas con IS_FRAUD (Cramér's V + chi²)

| Variable | Cramér's V | p-value | Significativa |
|---|---|---|---|
| `cuenta_origen` | **0.607** | 0.056 | ⚠️ Artefacto — ver sección 5.1 |
| `cuenta_destino` | **0.290** | ~0 | ✅ |
| `estado_tarjeta` | **0.226** | ~0 | ⚠️ Leakage — ver sección 5 |
| `destino_alto_riesgo` (riesgo_label) | 0.187 | ~0 | ✅ |
| `dispositivo_reconocido` (dispositivo_label) | 0.087 | ~0 | ✅ |
| `tipo_transaccion` | 0.072 | ~0 | ✅ |
| `metodo_autenticacion` | 0.032 | 0.036 | ✅ (débil) |
| resto (día semana, país, región...) | < 0.05 | > 0.05 | ❌ |

### 2.7 Análisis de interacciones entre variables (heatmaps + tablas)

Se analizaron las siguientes combinaciones:

1. `importe_tramo` × `dispositivo_reconocido` — dispositivos no reconocidos + importe alto = fraud rate ~30%
2. `is_night` × `numero_transacciones_ultima_hora`
3. `destino_alto_riesgo` × `importe_tramo`
4. `customer_country` × `operacion_pais`
5. `metodo_autenticacion` × `dispositivo_reconocido`
6. `numero_fraudes_ultimo_ano` × `destino_alto_riesgo`
7. `estado_tarjeta` × `importe_tramo`
8. `tiempo_desde_ultima_transaccion` × `numero_transacciones_ultima_hora`
9. `is_night` × `destino_alto_riesgo`
10. `tipo_transaccion` × `importe_tramo`

### 2.8 Revisión de variables adicionales (sección 8 del notebook)

Análisis individual de las variables no cubiertas en los bloques anteriores:

| Variable | Resultado |
|---|---|
| `estado_tarjeta`, `estado_cuenta` | Riesgo de leakage causal — **fuera del train** |
| `is_night`, `is_weekend` | Sin señal (diferencia < 1.5pp con la base) — **fuera** |
| `tipo_transaccion` | Transferencia 19.3% vs tarjeta 13.6% — **dentro** |
| `metodo_autenticacion` | Señal débil pero coherente — **candidata opcional** |
| `numero_transacciones_ultima_hora` | Correlación ≈ 0 — **fuera** |
| `saldo_actual` vs `saldo_medio_30_dias` | Correlación 0.98 entre sí — **se queda `saldo_actual`, fuera `saldo_medio_30_dias`** |
| `limite_importe_transacciones` | 5 valores, fraud rate plano al 15% — **fuera** |

---

## 3. Estado del notebook

### 3.1 Completado ✅

- ✅ Análisis univariante, bivariante, correlaciones e interacciones
- ✅ Outliers en variables numéricas (boxplots + tabla IQR + escala logarítmica)
- ✅ Distribuciones de saldo, volumen, antigüedad, tiempo desde última transacción
- ✅ Visualización del desbalanceo de clases
- ✅ Investigación de `cuenta_origen` — resuelta, ver sección 5.1
- ✅ Análisis cross-border: 10% de operaciones en país diferente al cliente → 26.7% vs 14.0%
- ✅ Análisis de `tenure` y `antiguedad_tarjeta_dias`
- ✅ Análisis de `numero_fraudes_ultimo_ano` — resuelta, ver sección 5.2
- ✅ Revisión de variables adicionales (sección 8)
- ✅ 16 celdas interpretativas (hallazgos por bloque de análisis)
- ✅ **Sección 9 — Conclusiones del EDA** (variables descartadas, señal confirmada, anomalías, baseline AUC)
- ✅ **Sección 10 — Recomendaciones de Feature Engineering y Modelado**

### 3.2 Coordinación con el equipo (para más adelante)

- **Acordar con Full Stack el formato de salida del modelo** — esquema JSON/API para las predicciones. Se hará cuando el primer modelo esté entrenado.
- **Acordar con Red Team qué variables atacarán en Ronda 2** — preparar un documento de una página con el feature set usado, no el EDA completo.

---

## 4. Decisiones sobre variables — resumen

### Dentro del feature set

| Variable | Señal |
|---|---|
| `destino_alto_riesgo` | 33.6% vs 12.8% — señal más limpia del dataset |
| `dispositivo_reconocido` | No reconocido: 22.7% vs reconocido: 13.9% |
| `tipo_transaccion` | Transferencia: 19.3% vs tarjeta: 13.6% |
| `importe_transaccion` | Quintil superior: 21.9%. Fuerte en combinación |
| `fraudes_prev_capped` | Derivada de `numero_fraudes_ultimo_ano` con cap en 3 |
| `cross_border` (derivada) | Operación fuera del país del cliente: 26.7% vs 14.0% |
| `saldo_actual` | Contexto de la transacción, relevante para ratios |
| `importe_medio_mensual` | Correlación débil (0.033) pero útil para ratios |
| `metodo_autenticacion` | Candidata opcional — señal débil pero coherente |
| `numero_pin_disponibles` | Valor 0 → 23.5% fraud rate — a confirmar en FE |
| Variables de volumen y transferencias | Señal individual débil, pueden aportar en combinación |

### Fuera del feature set

| Variable | Motivo |
|---|---|
| `id_cliente`, `id_cuenta`, `id_tarjeta`, `id_transaccion` | Identificadores |
| `IMPACTO_FRAUDE` | **Leakage total** — correlación 0.89 con IS_FRAUD, siempre 0 en no-fraudes |
| `cuenta_origen` | Artefacto de alta cardinalidad — sin señal real (AUC idéntico con/sin ella) |
| `estado_tarjeta`, `estado_cuenta` | **Riesgo de leakage causal** en dataset sintético |
| `saldo_medio_30_dias` | Redundante con `saldo_actual` (correlación 0.98) |
| `is_night`, `is_weekend` | Sin señal |
| `numero_transacciones_ultima_hora` | Sin señal |
| `veces_superar_limite_7_dias` | Sin señal |
| `limite_importe_transacciones` | Sin señal |
| `identificador_dispositivo_fingerprint`, `direccion_ip_origen`, `geolocalizacion`, `cuenta_destino` | Identificadores o requieren ingeniería compleja fuera del alcance de Ronda 1 |
| `fecha_hora`, `fecha_creacion_tarjeta` | Fechas en bruto — usar solo derivadas |
| `numero_fraudes_ultimo_ano` | Sustituida por `fraudes_prev_capped` |

### Features derivadas a construir en FE

```python
df['fraudes_prev_capped'] = df['numero_fraudes_ultimo_ano'].clip(upper=3)
df['cross_border'] = (df['operacion_pais'] != df['customer_country']).astype(int)
df['ratio_importe_saldo'] = df['importe_transaccion'] / (df['saldo_actual'] + 1)
df['ratio_importe_media'] = df['importe_transaccion'] / (df['importe_medio_mensual'] + 1)
```

---

## 5. Anomalías del dataset — análisis y decisiones tomadas

### 5.1 `cuenta_origen` — falsa señal de alta cardinalidad

**Qué se detectó:** Cramér's V = 0.61 con `IS_FRAUD`, pero p-value = 0.056 (no significativo). Había 143 cuentas con fraud rate del 100%.

**Diagnóstico:** Tras investigación, no hay señal real. Las 143 cuentas "100% fraude" son casi todas cuentas con 1-2 transacciones — con tan pocas observaciones, es estadísticamente trivial tener fraud rate del 100% o 0%. El Cramér's V alto es un artefacto de tener ~3.548 categorías únicas con muy pocas observaciones por categoría; eso infla artificialmente el estadístico. Prueba definitiva: el AUC del modelo **no cambia** al incluir o excluir `cuenta_origen` (0.706 sin ella, 0.705 con ella).

**Decisión:** `cuenta_origen` se **descarta** como feature. No hay que hacer nada especial en el código — simplemente no incluirla en la lista de features del modelo.

---

### 5.2 `numero_fraudes_ultimo_ano` — rotura aparente en valores altos

**Qué se detectó:** La fraud rate sube de forma lógica de 0 a 3 fraudes previos (14.4% → 16.1% → 19.4% → 23.4%), pero luego cae de forma extraña: 4 fraudes → 11.1%, 5 fraudes → 10.0%, 6 fraudes → 0.0%. Parecía un error del generador sintético.

**Diagnóstico:** Es ruido estadístico puro de muestra pequeña. Los intervalos de confianza al 95% lo confirman:

| Valor | n | Fraud rate | IC 95% |
|---|---|---|---|
| 0 | 6.791 | 14.4% | [13.6%, 15.2%] |
| 1 | 2.240 | 16.1% | [14.7%, 17.7%] |
| 2 | 685 | 19.4% | [16.6%, 22.5%] |
| 3 | 201 | 23.4% | [18.1%, 29.7%] |
| 4 | 72 | 11.1% | [5.7%, **20.4%**] ← incluye la tasa base |
| 5 | 10 | 10.0% | [1.8%, **40.4%**] ← casi sin valor |
| 6 | 1 | 0.0% | [0.0%, **79.3%**] ← irrelevante |

El 99.2% de los datos están en valores 0-3 (9.917 de 10.000). El aparente descenso en 4-6 no es real — los intervalos de confianza de esos valores incluyen perfectamente la tasa base del 15.26%. No hay que regenerar el dataset.

**Decisión:** Usar la variable con un **cap en 3**:

```python
df['fraudes_prev_capped'] = df['numero_fraudes_ultimo_ano'].clip(upper=3)
```

Esta nueva columna captura la tendencia real y creciente (0→1→2→3), sin que el modelo intente aprender de 83 registros con alta varianza.

---

## 6. Próximo paso — Feature Engineering

El EDA está cerrado. El siguiente notebook es `FE.ipynb` (feature engineering). No existe aún — hay que crearlo.

### Lo que hay que implementar (decidido, solo ejecutar)

```python
# 1. No incluir cuenta_origen en el feature set (no hace falta dropearla)
# Documentar la decisión en una celda Markdown.

# 2. Cap en numero_fraudes_ultimo_ano
df['fraudes_prev_capped'] = df['numero_fraudes_ultimo_ano'].clip(upper=3)

# 3. Flag cross-border
df['cross_border'] = (df['operacion_pais'] != df['customer_country']).astype(int)

# 4. Ratios de importe
df['ratio_importe_saldo'] = df['importe_transaccion'] / (df['saldo_actual'] + 1)
df['ratio_importe_media'] = df['importe_transaccion'] / (df['importe_medio_mensual'] + 1)
```

### Lo que hay que decidir en equipo

- **Selección final de features**: cuáles de las variables de señal débil entran (volumen, transferencias 7 días, `metodo_autenticacion`, `numero_pin_disponibles`). Requiere criterio sobre el trade-off complejidad/ganancia.
- **Estrategia de encoding**: One-Hot para categoriales de pocas clases; Target Encoding con smoothing para `customer_country` u `operacion_pais` si se usan.
- **Umbral de decisión del modelo**: ajustar según el trade-off precisión/recall que defina negocio.

### Recomendaciones de modelado (ver sección 10 del notebook para detalle)

- **Métricas**: AUC-ROC principal, F1 secundario. No usar accuracy.
- **Desbalanceo**: `class_weight='balanced'` o SMOTE.
- **Baseline**: Random Forest (AUC 0.706 ya conseguido sin FE).
- **Candidato principal**: XGBoost o LightGBM.
- **Validación**: 5-fold cross-validation estratificada.
