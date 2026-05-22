# Contexto del EDA — NovaPay Operación Centinela (Ronda 1)

> Documento de traspaso generado el 22/05/2026.
> Autor del análisis hasta este punto: Juan Antonio (Data Science — Blue Team, Grupo 2).
> Este documento está pensado para que un compañero pueda retomar el trabajo sin perder contexto.

> **Objetivo de esta sesión:** dejar el EDA terminado y bien presentado. El modelado y el feature engineering son el siguiente bloque, y lo abordaremos juntos una vez el EDA esté cerrado — es el flujo natural: primero entender bien los datos, luego decidir cómo transformarlos y qué modelo usar.

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

## 3. Lo que falta por hacer

> Todo lo de esta sección es para cerrar el EDA. Una vez esté hecho, el notebook queda listo para entregarlo como documentación y para que el modelado arranque con una base sólida.

### 3.1 Análisis pendiente

- [ ] **Outliers en variables numéricas continuas** — No hay análisis de outliers para `saldo_actual`, `importe_transaccion`, `volumen_saliente/entrante_30_dias`, `tiempo_desde_ultima_transaccion`, etc. Hace falta al menos boxplots o IQR analysis, especialmente relevante para el modelo.
- [ ] **Distribución de variables de cuenta/saldo no exploradas** — `saldo_actual`, `saldo_medio_30_dias`, `volumen_entrante_30_dias`, `volumen_saliente_30_dias`, `importe_medio_mensual`, `desviacion_estandar_mensual`, `tenure`, `tiempo_desde_ultima_transaccion` no tienen histograma ni KDE propio con comparativa fraude/no fraude.
- [ ] **Visualización explícita del desbalanceo de clases** — Solo se ve en `.describe()` (media IS_FRAUD = 0.1526). Hace falta un gráfico claro (pie o barplot) que muestre las proporciones, y una reflexión sobre cómo afectará al modelado (class_weight, SMOTE, métricas a usar...).
- [ ] **Investigar `cuenta_origen`** — Identificar si hay cuentas con fraud rate del 100% o próximo. Si existen, ¿son pocas cuentas? ¿El dataset las incluyó con intención? Esto puede afectar al modelo (data leakage si esas cuentas aparecen en train y test).
- [ ] **Análisis geográfico cross-border** — El heatmap `customer_country × operacion_pais` está generado pero no interpretado. Falta calcular explícitamente qué % de transacciones son cross-border (país del cliente ≠ país de la operación) y si eso eleva el fraude.
- [ ] **Análisis de `tenure` y `antiguedad_tarjeta_dias`** — No se han explorado en relación con el fraude. Los clientes nuevos o con tarjetas recién creadas pueden ser más vulnerables.
- [ ] **Análisis de `tiempo_desde_ultima_transaccion`** — La unidad parece ser segundos (max ~15M ≈ 174 días), pero no está documentado. Falta explorar si transacciones muy rápidas (tiempo bajo) o muy espaciadas tienen mayor fraude rate.

### 3.2 Features derivadas a documentar en el EDA (solo apuntar la idea, no implementar)

El EDA debe terminar con una lista clara de qué variables nuevas tendría sentido crear. No hace falta construirlas aquí — eso va en el notebook de feature engineering. Lo que sí conviene dejar escrito en el EDA es el razonamiento detrás de cada una, para que cuando lleguemos al modelado ya tengamos la lógica pensada:

- **Flag cross-border** — `operacion_pais != customer_country` → ¿opera desde un país diferente al del cliente?
- **Ratio importe / saldo_actual** — ¿La transacción consume un % alto del saldo disponible?
- **Ratio volumen_saliente / volumen_entrante** — ¿La cuenta solo "saca" dinero?
- **Importe vs media histórica** — `importe_transaccion / importe_medio_mensual` — ¿es una transacción inusualmente grande para ese cliente?
- **Velocidad** — combinar `numero_transacciones_ultima_hora` y `tiempo_desde_ultima_transaccion` para detectar ráfagas de actividad

### 3.3 Organización del notebook

- [ ] **Añadir celdas Markdown de título/sección** — El notebook no tiene estructura visual. Necesita secciones tipo: `## 1. Carga y limpieza`, `## 2. Análisis univariante`, `## 3. Análisis bivariante`, `## 4. Correlaciones`, `## 5. Interacciones`, `## 6. Conclusiones y features seleccionadas`
- [ ] **Corregir encoding** — En alguna celda aparece "CategÃ³ricas" en vez de "Categóricas". Hay un problema de encoding UTF-8 en una o dos celdas que hay que arreglar.
- [ ] **Eliminar o consolidar redundancias** — Hay variables creadas con `.copy()` en varios sitios (`df_plot`, `df_inter`, `tabla_resumen`...) que deberían unificarse o limpiarse para no confundir al lector.

### 3.4 Interpretación y conclusiones

- [ ] **Añadir texto interpretativo bajo cada gráfico/bloque** — Actualmente el notebook es solo código + outputs sin explicar qué significa lo que se ve. Cada sección debería tener una celda Markdown de 2-4 líneas con el hallazgo clave.
- [ ] **Celda de conclusiones finales del EDA** — Necesita un bloque resumen al final con: (1) variables más relevantes identificadas, (2) patrones de fraude detectados, (3) advertencias/caveats sobre el dataset sintético, (4) recomendaciones para el modelo.
- [ ] **Lista de features recomendadas para el modelo** — Resultado accionable del EDA: qué variables incluir, cuáles descartar (IDs, fechas crudas, geolocalización en texto), y cuáles derivar.

### 3.5 Coordinación con el equipo

Esto no es para esta sesión, pero conviene tenerlo en mente mientras se cierra el EDA:

- **Acordar con Full Stack el formato de salida del modelo** — El modelo debe exponer sus predicciones (probabilidad de fraude + etiqueta) en un formato que Full Stack pueda consumir. Definir esquema JSON/API. Lo haremos cuando el EDA esté cerrado y sepamos qué variables usa el modelo.
- **Acordar con Red Team (Ciberseguridad) qué variables atacarán en Ronda 2** — Saber qué features va a manipular el Red Team ayuda a no depender solo de ellas en el modelo. También lo abordaremos en el siguiente bloque.

---

## 4. Variables destacadas — resumen provisional para cerrar el EDA

Esta tabla es el resultado que debe quedar documentado al final del EDA — una referencia clara para cuando arranquemos con el feature engineering y el modelado. No hay que implementar nada ahora, solo asegurarse de que el análisis que ya está hecho la justifica y de completar lo que falta para poder rellenar los huecos.

Basado en lo analizado hasta ahora, las señales más prometedoras son:

**Fuertes:**
- `destino_alto_riesgo` (correlación 0.19, Cramér's V 0.19 — la más clara)
- `estado_tarjeta` (Cramér's V 0.23 — significativo)
- `cuenta_destino` (Cramér's V 0.29 — ojo con cardinalidad alta, 500 valores únicos)
- `dispositivo_reconocido` (correlación -0.09, Cramér's V 0.09)
- `importe_transaccion` (correlación 0.08 — señal débil pero combinada con otros mejora)

**Moderadas:**
- `tipo_transaccion`
- `metodo_autenticacion`
- `numero_fraudes_ultimo_ano`
- `numero_transacciones_ultima_hora`
- `veces_superar_limite_7_dias`

**A descartar o tratar con cuidado:**
- `cuenta_origen` — Cramér's V muy alto pero p-value no significativo; investigar antes de usar
- IDs (`id_cliente`, `id_cuenta`, `id_tarjeta`, `id_transaccion`) — no usar como features
- `direccion_ip_origen`, `geolocalizacion` — texto libre, requieren parsing si se quieren usar
- `fecha`, `fecha_hora`, `fecha_creacion_tarjeta` — usar solo features derivadas (hora, dia_semana, etc.)
