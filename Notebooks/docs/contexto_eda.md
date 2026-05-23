# Contexto del EDA — NovaPay Operación Centinela (Ronda 1)

> Documento de traspaso. Última actualización: **23/05/2026**.
> Autor del análisis: Juan Antonio (Data Science — Blue Team, Grupo 2).
> Este documento está pensado para que un compañero pueda retomar el trabajo sin perder contexto.

> **Estado actual:** el EDA está prácticamente cerrado. El notebook ordenado (`EDA_ordenado.ipynb`) tiene estructura completa, análisis hecho e interpretaciones escritas. **El único paso que queda antes de pasar al modelado** es tratar dos anomalías del dataset que ya están diagnosticadas — están explicadas en detalle en la sección 5 de este documento. Es trabajo concreto y acotado, no hay que tomar decisiones de diseño: está todo decidido, solo hay que implementarlo.

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
- El dataset está **desbalanceado** (~6,5:1 legítimas/fraudulentas)

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
- **Fraud rate por mes:** ligeramente más bajo en noviembre (~12%) y en febrero/abril (~15%), pero sin patrón fuerte.
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
| `cuenta_origen` | **0.607** | 0.056 | ⚠️ NO (casi) |
| `cuenta_destino` | **0.290** | ~0 | ✅ |
| `estado_tarjeta` | **0.226** | ~0 | ✅ |
| `destino_alto_riesgo` (riesgo_label) | 0.187 | ~0 | ✅ |
| `dispositivo_reconocido` (dispositivo_label) | 0.087 | ~0 | ✅ |
| `tipo_transaccion` | 0.072 | ~0 | ✅ |
| `metodo_autenticacion` | 0.032 | 0.036 | ✅ (débil) |
| resto (día semana, país, región...) | < 0.05 | > 0.05 | ❌ |

> ⚠️ **Nota importante sobre `cuenta_origen`:** El Cramér's V altísimo (0.61) pero p-value no significativo (0.056) sugiere que hay muy pocas cuentas que concentran casi todo el fraude. Esto puede ser un artefacto del dataset sintético (cuentas "spammer") o una señal real muy útil. **Requiere investigación antes de usar como feature.**

- Barplot de Cramér's V por variable

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

---

## 3. Lo que ya está hecho

### 3.1 Análisis completado ✅

- ✅ Outliers en variables numéricas (boxplots + tabla IQR + escala logarítmica)
- ✅ Distribuciones de saldo, volumen, antigüedad, tiempo desde última transacción (fraude vs no fraude)
- ✅ Visualización del desbalanceo de clases con reflexión sobre impacto en el modelado
- ✅ Investigación de `cuenta_origen` — resuelta, ver sección 5
- ✅ Análisis cross-border: 10% de operaciones en país diferente al cliente → fraud rate 26.7% vs 14.0%
- ✅ Análisis de `tenure` y `antiguedad_tarjeta_dias`
- ✅ Análisis de `tiempo_desde_ultima_transaccion` (confirmado en segundos, max ~174 días)
- ✅ Análisis de `numero_fraudes_ultimo_ano` — resuelta, ver sección 5

### 3.2 Organización del notebook ✅

- ✅ `EDA_ordenado.ipynb` creado con secciones Markdown, comentarios de código e interpretaciones
- ✅ Estructura: Setup → Carga → Univariante → Bivariante → Correlaciones → Interacciones
- ✅ 16 celdas interpretativas añadidas (una por bloque de análisis)

### 3.3 Pendiente — conclusiones finales

Lo que queda de EDA puro lo cerrará Juan Antonio cuando vuelva:

- [ ] **Celda de conclusiones finales** — resumen de variables relevantes, patrones detectados, caveats del dataset sintético y recomendaciones para el modelo
- [ ] **Lista definitiva de features para el modelo** — qué incluir, qué descartar y qué derivar

Estas dos celdas las escribe Juan Antonio porque requieren criterio de diseño de modelo. **No tocar.**

### 3.4 Coordinación con el equipo (para más adelante)

- **Acordar con Full Stack el formato de salida del modelo** — esquema JSON/API para las predicciones. Se hará cuando el EDA esté cerrado.
- **Acordar con Red Team qué variables atacarán en Ronda 2** — para no depender de features atacables. También en el siguiente bloque.

---

## 4. Variables destacadas — resumen para el modelado

**Fuertes:**
- `destino_alto_riesgo` — fraud rate 33.6% vs 12.8%. La señal más limpia del dataset.
- `estado_tarjeta` — tarjetas robadas/extraviadas/bloqueadas: 36-41% de fraude vs 12% en activas.
- `cuenta_destino` (Cramér's V 0.29 — ojo con cardinalidad alta, ~500 valores únicos)
- `dispositivo_reconocido` — no reconocido: 22.7% fraude vs 13.9% reconocido.
- `cross_border` (feature derivada) — país de operación ≠ país del cliente: 26.7% vs 14.0%.

**Moderadas:**
- `importe_transaccion` — top quintil al 21.9%. Señal débil sola, fuerte en combinación.
- `tipo_transaccion` — transferencias: 19.3% vs tarjeta: 13.6%.
- `metodo_autenticacion` — firma/contactless levemente peor que 3DS.
- `numero_transacciones_ultima_hora` y `veces_superar_limite_7_dias` — crecientes con el fraude.
- `numero_fraudes_ultimo_ano` — útil en rango 0-3; **usar con cap en 3** (ver sección 5.2).

**Descartar:**
- `cuenta_origen` — descartada. Falsa señal de alta cardinalidad. Ver sección 5.1.
- IDs (`id_cliente`, `id_cuenta`, `id_tarjeta`, `id_transaccion`) — nunca como features.
- `direccion_ip_origen`, `geolocalizacion` — texto libre sin parsear, no usar directamente.
- `fecha`, `fecha_hora`, `fecha_creacion_tarjeta` — usar solo derivadas (hora, dia_semana, etc.).

**Features derivadas recomendadas para el FE:**
- `flag_cross_border` — `operacion_pais != customer_country`
- `ratio_importe_saldo` — `importe_transaccion / saldo_actual`
- `ratio_volumen_saliente_entrante` — `volumen_saliente_30_dias / volumen_entrante_30_dias`
- `ratio_importe_media` — `importe_transaccion / importe_medio_mensual`
- `fraudes_prev_capped` — `min(numero_fraudes_ultimo_ano, 3)`

---

## 5. Anomalías del dataset — análisis y decisiones tomadas

> Esta sección documenta dos anomalías detectadas en el dataset sintético, su diagnóstico completo y la decisión sobre cómo tratarlas. **El siguiente paso concreto es implementar estas dos correcciones** — están al 100% resueltas en análisis, solo falta escribir el código. Se implementarán en el notebook de feature engineering, no en el EDA.

### 5.1 `cuenta_origen` — falsa señal de alta cardinalidad

**Qué se detectó:** Cramér's V = 0.61 con `IS_FRAUD`, pero p-value = 0.056 (no significativo). Había 143 cuentas con fraud rate del 100%.

**Diagnóstico:** Tras investigación, no hay señal real. Las 143 cuentas "100% fraude" son casi todas cuentas con 1-2 transacciones — con tan pocas observaciones, es estadísticamente trivial tener fraud rate del 100% o 0%. El Cramér's V alto es un artefacto de tener ~3.548 categorías únicas con muy pocas observaciones por categoría; eso infla artificialmente el estadístico. Prueba definitiva: el AUC del modelo **no cambia** al incluir o excluir `cuenta_origen` (0.706 sin ella, 0.705 con ella).

**Decisión:** `cuenta_origen` se **descarta** como feature. No hay que hacer nada especial en el código — simplemente no incluirla en la lista de features del modelo. La nota de "investigar antes de usar" que aparecía en versiones anteriores de este documento queda cerrada.

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

**Decisión:** Usar la variable con un **cap en 3**. En el notebook de feature engineering, añadir:

```python
df['fraudes_prev_capped'] = df['numero_fraudes_ultimo_ano'].clip(upper=3)
```

Esta nueva columna captura la tendencia real y creciente (0→1→2→3), sin que el modelo intente aprender de 83 registros con alta varianza. La columna original `numero_fraudes_ultimo_ano` se puede mantener pero no usar directamente como feature.

---

## 6. Próximo paso — lo que hay que hacer ahora

> Esto es lo que le queda al compañero que retome el trabajo. Está todo decidido — solo hay que ejecutarlo.

**Tarea única: implementar las dos correcciones en el notebook de feature engineering.**

El EDA está cerrado. El siguiente notebook es el de **feature engineering**, que aún no existe. Hay que crearlo (`FE.ipynb` o `feature_engineering.ipynb`) y que arranque con estas dos líneas antes de cualquier otra transformación:

```python
# 1. Eliminar cuenta_origen del conjunto de features (no usar en el modelo)
# No hace falta dropearla del dataframe, simplemente no incluirla en la lista de features.
# Documenta la decisión en una celda Markdown.

# 2. Capear numero_fraudes_ultimo_ano en 3
df['fraudes_prev_capped'] = df['numero_fraudes_ultimo_ano'].clip(upper=3)
```

Después de estas dos correcciones, el notebook de FE puede continuar con el resto de features derivadas listadas en la sección 4 (cross-border, ratios de importe/saldo, etc.). Eso ya lo haremos con Juan Antonio cuando vuelva — es donde empieza el criterio de diseño del modelo.
