# NovaPay ML — Informe para Equipo Ciberseguridad
**Operación Centinela · Blue Team, Grupo 2**  
**Versión del modelo: v1 activo · Mayo 2026**

> Este documento está escrito específicamente para el equipo de Ciberseguridad.  
> El objetivo es que entendáis exactamente qué detecta nuestro modelo, qué patrones aprendió  
> y cómo construir transacciones fraudulentas realistas que lo pongan a prueba.

---

## 1. Cómo decide el modelo si una transacción es fraude

El modelo no ve los campos que os envía la app directamente. Primero los transforma en **67 indicadores calculados** (features) y luego decide. La cadena completa es:

```
Vuestro JSON  →  67 features calculadas  →  LightGBM + XGBoost  →  prob_fraud  →  is_fraud
```

El **umbral de decisión es 0.3144**: cualquier transacción con `prob_fraud >= 0.3144` se marca como fraude. Esto significa que no hace falta ser "muy fraudulento" para que salte la alarma — con superar ese 31% de probabilidad es suficiente.

**Cuanto más combináis patrones sospechosos en una misma transacción, más sube la probabilidad.**

---

## 2. Los 8 patrones que el modelo conoce

Estos son los patrones de fraude que aprendió el modelo durante el entrenamiento. Son los mismos que usó el generador de datos para marcar transacciones como fraudulentas, así que el modelo los reconoce con alta fiabilidad.

---

### Patrón 1 — País diferente al del cliente
**Campo:** `operacion_pais` ≠ `customer_country`  
**Frecuencia en fraudes del dataset:** 55%

El modelo calcula internamente `cross_border = 1` cuando el país de la operación no coincide con el país del cliente. Es uno de los indicadores más fuertes.

```json
"customer_country": "ES",
"operacion_pais": "RO"
```
> Combinadlo con `operacion_region` que tampoco coincida con `customer_region` para activar también `cross_region`.

---

### Patrón 2 — Dispositivo no reconocido
**Campo:** `dispositivo_reconocido = 0`  
**Frecuencia en fraudes:** 60%

El modelo presta mucha atención a si el dispositivo desde el que se hace la transacción es conocido o no. En el dataset legítimo, ~85% de transacciones vienen de dispositivos reconocidos. Ver `dispositivo_reconocido=0` en combinación con otros factores es muy sospechoso.

```json
"dispositivo_reconocido": 0,
"identificador_dispositivo_fingerprint": "dispositivo-nuevo-desconocido"
```

---

### Patrón 3 — Ráfaga de transacciones (burst)
**Campos:** `numero_transacciones_ultima_hora` y `tiempo_desde_ultima_transaccion`  
**Frecuencia en fraudes:** 45%

El modelo tiene features específicas para detectar ráfagas. Los umbrales exactos que activan las alarmas internas:

| Feature interna | Se activa cuando... |
|----------------|---------------------|
| `burst_rapido` | >5 txns en la última hora AND última txn hace < 5 min (300s) |
| `alta_velocidad` | >3 txns en la última hora OR última txn hace < 30s |
| `txn_por_minuto` | Valor alto = muchas txns en poco tiempo |

```json
"numero_transacciones_ultima_hora": 8,
"tiempo_desde_ultima_transaccion": 45
```

> Para ataques batch (`POST /predict/batch`), podéis enviar muchas transacciones del mismo cliente con `tiempo_desde_ultima_transaccion` bajo, simulando el efecto de ráfaga.

---

### Patrón 4 — Importe cerca del límite de la tarjeta
**Campos:** `importe_transaccion`, `limite_importe_transacciones`  
**Frecuencia en fraudes:** 40%

El modelo calcula `txn_vs_limit_pct = importe / límite`. Cuando este ratio supera ~0.85 (85% del límite), es sospechoso. Aún más si el importe es un número redondo (100, 500, 1000, 2000...) porque el modelo tiene una feature específica para eso:

- `high_ratio_redondeado = 1` cuando: `importe/limite > 0.85` AND `importe % 100 < 1`

```json
"importe_transaccion": 1900.00,
"limite_importe_transacciones": 2000.00
```
> `txn_vs_limit_pct = 0.95` → muy sospechoso.  
> Si ponéis `importe = 2000.00` con `limite = 2000.00` → `high_ratio_redondeado = 1` también.

---

### Patrón 5 — Sin intentos de PIN disponibles
**Campo:** `numero_pin_disponibles = 0`  
**Frecuencia en fraudes:** 35%

Indica que el PIN ha sido bloqueado o agotado. En el dataset normal, solo ~2% de transacciones llegan a este estado.

```json
"numero_pin_disponibles": 0
```

---

### Patrón 6 — Método de autenticación débil o anómalo
**Campo:** `metodo_autenticacion`  
**Frecuencia en fraudes:** 35%

Los métodos más sospechosos son `"firma"` y `"3DS"`. El modelo aprende la frecuencia de cada método en los datos de entrenamiento (encoding de frecuencia): métodos inusuales tienen menor frecuencia codificada, lo que sube la sospecha.

```json
"metodo_autenticacion": "firma"
```

---

### Patrón 7 — Destino de alto riesgo
**Campo:** `destino_alto_riesgo = 1`  
**Frecuencia en fraudes:** 30%

Transacciones hacia cuentas marcadas como destinos de riesgo.

```json
"destino_alto_riesgo": 1
```

---

### Patrón 8 — Transacciones muy seguidas
**Campo:** `tiempo_desde_ultima_transaccion < 55`  
**Frecuencia en fraudes:** 30%

Menos de 55 segundos desde la última transacción es inusual. Cuanto menor sea este valor, mayor es la sospecha. Combinado con `numero_transacciones_ultima_hora` alto, activa el burst detector.

```json
"tiempo_desde_ultima_transaccion": 20
```

---

## 3. Features compuestas — las señales más fuertes

El modelo tiene 4 indicadores que combinan varios patrones a la vez. Estos tienen **mucho más peso** que los patrones individuales porque son señales de alta confianza:

| Feature compuesta | Se activa cuando | Por qué es fuerte |
|-------------------|------------------|-------------------|
| `foreign_unknown_device` | `cross_border=1` AND `dispositivo_reconocido=0` | País extranjero + dispositivo nuevo = perfil de robo de tarjeta |
| `night_velocity` | `is_night=1` AND `alta_velocidad=1` | Ráfaga nocturna = muy sospechoso |
| `high_ratio_redondeado` | `importe/limite > 0.85` AND importe redondo | Intentar vaciar la cuenta con importes exactos |
| `burst_rapido` | >5 txns última hora AND <5 min desde última | Patrón clásico de ataque automatizado |

**Para maximizar `prob_fraud`, lo más efectivo es activar `foreign_unknown_device` + `burst_rapido` a la vez.**

---

## 4. Indicadores financieros que el modelo calcula

Más allá de los patrones de comportamiento, el modelo también mira el estado financiero de la cuenta. Estos ratios tienen peso en la predicción:

| Feature calculada | Fórmula | Valor sospechoso |
|-------------------|---------|-----------------|
| `txn_vs_balance_pct` | importe / saldo_actual | Alto (> 0.8) |
| `outflow_inflow_ratio` | vol_saliente / vol_entrante | > 1.5 (sale más de lo que entra) |
| `net_flow_30d` | vol_entrante − vol_saliente | Muy negativo |
| `txn_severity` | importe × txns_última_hora | Alto |
| `limite_breach_rate` | veces_superar_limite_7d / 7 | > 0 |
| `txn_intensity` | txns_última_hora / (tiempo_última + 1) | Alto |
| `ratio_actividad_cliente` | txns_última_hora / media_diaria_cliente | Mucho mayor que 1 |
| `diff_importe_cliente` | \|importe − media_cliente\| / media_cliente | Alto (importe muy distinto a lo habitual) |

**Ejemplo de perfil de cuenta que maximiza estos ratios:**
```json
"saldo_actual": 300.00,
"importe_transaccion": 280.00,
"volumen_entrante_30_dias": 500.00,
"volumen_saliente_30_dias": 2800.00,
"veces_superar_limite_7_dias": 3,
"importe_medio_mensual": 50.00
```
> `txn_vs_balance_pct = 0.93`, `outflow_inflow_ratio = 5.6`, `diff_importe_cliente = 4.6` → muy sospechoso.

---

## 5. Lo que el modelo NO mira (campos ignorados)

Estos campos se descartan antes de que el modelo los vea. **No sirve de nada manipularlos:**

| Campo | Por qué se ignora |
|-------|------------------|
| `estado_cuenta` | Poca señal discriminativa; puede reflejar consecuencia del fraude (leakage) |
| `estado_tarjeta` | Mismo motivo |
| `id_cliente`, `id_cuenta`, `id_tarjeta`, `id_transaccion` | Identificadores únicos, no aportan señal |
| `cuenta_origen` | ID de cuenta, no aporta señal directa |
| `direccion_ip_origen` | Se descarta (se usa `operacion_pais` y `cross_border` en su lugar) |
| `geolocalizacion` | Se descarta |
| `fecha_creacion_tarjeta` | Se descarta (se usa `antiguedad_tarjeta_dias`) |

---

## 6. Cómo construir una transacción fraudulenta — ejemplo completo

Aquí tenéis un JSON de ejemplo que activa múltiples patrones a la vez y debería tener `prob_fraud` muy alta:

```json
{
  "id_transaccion"                        : "ataque-ciber-001",
  "id_cliente"                            : "cliente-victima-001",
  "tipo_cliente"                          : "persona",
  "edad_cliente"                          : 45,
  "customer_country"                      : "ES",
  "customer_region"                       : "Centro",
  "tenure"                                : 730,
  "importe_medio_mensual"                 : 200.00,
  "desviacion_estandar_mensual"           : 50.00,
  "media_transacciones_al_dia"            : 2.0,
  "numero_fraudes_ultimo_ano"             : 0,

  "id_cuenta"                             : "cuenta-001",
  "cuenta_origen"                         : "ES20000000000000000001",
  "estado_cuenta"                         : "activa",
  "saldo_actual"                          : 400.00,
  "saldo_medio_30_dias"                   : 2000.00,
  "volumen_entrante_30_dias"              : 300.00,
  "volumen_saliente_30_dias"              : 3500.00,
  "numero_transferencias_recibidas_7_dias": 1,
  "numero_transferencias_enviadas_7_dias" : 8,

  "id_tarjeta"                            : "tarjeta-001",
  "estado_tarjeta"                        : "activa",
  "fecha_creacion_tarjeta"                : "2023-01-01",
  "antiguedad_tarjeta_dias"               : 365,
  "limite_importe_transacciones"          : 2000.00,
  "veces_superar_limite_7_dias"           : 2,

  "tipo_transaccion"                      : "tarjeta",
  "fecha_hora"                            : "2026-05-26 03:15:00",
  "is_night"                              : 1,
  "is_weekend"                            : 0,
  "tiempo_desde_ultima_transaccion"       : 25,
  "numero_transacciones_ultima_hora"      : 7,
  "importe_transaccion"                   : 1900.00,
  "metodo_autenticacion"                  : "firma",
  "numero_pin_disponibles"                : 0,
  "identificador_dispositivo_fingerprint" : "dispositivo-desconocido-xyz",
  "dispositivo_reconocido"                : 0,
  "operacion_pais"                        : "RO",
  "operacion_region"                      : "Este",
  "direccion_ip_origen"                   : "185.220.101.1",
  "geolocalizacion"                       : "44.4268,26.1025",
  "cuenta_destino"                        : "RO49AAAA1B31007593840000",
  "destino_alto_riesgo"                   : 1
}
```

**Patrones activos en este JSON:**

| Patrón | Valor | Feature activada |
|--------|-------|-----------------|
| País extranjero | `operacion_pais=RO` ≠ `customer_country=ES` | `cross_border=1` |
| Dispositivo desconocido | `dispositivo_reconocido=0` | — |
| **COMBINADO** | Los dos anteriores | **`foreign_unknown_device=1`** ← señal muy fuerte |
| Ráfaga nocturna | `is_night=1` + 7 txns + 25s | **`night_velocity=1`**, `burst_rapido=1` |
| Importe cerca del límite | 1900 / 2000 = 95% | `txn_vs_limit_pct=0.95` |
| PIN agotado | `numero_pin_disponibles=0` | — |
| Autenticación débil | `metodo_autenticacion=firma` | — |
| Destino alto riesgo | `destino_alto_riesgo=1` | — |
| Salida masiva | `vol_saliente >> vol_entrante` | `outflow_inflow_ratio=11.7` |
| Importe inusual | 1900 vs media de 200 | `diff_importe_cliente` alto |

---

## 7. Estrategia para ataques — Ronda 1

### Ataques individuales (`POST /predict`)

Para fraudes individuales que el modelo detecte claramente, combinad al menos **3-4 patrones** del apartado 2. Uno solo puede no ser suficiente para superar el umbral 0.3144.

**Combinaciones de mayor impacto (de mayor a menor):**
1. `foreign_unknown_device` + `burst_rapido` + importe alto
2. `is_night` + `burst_rapido` + `cross_border`
3. `dispositivo_reconocido=0` + importe > 85% límite + `destino_alto_riesgo`

### Ataques masivos (`POST /predict/batch`)

Podéis enviar un array de transacciones en una sola llamada:

```
POST http://localhost:8000/predict/batch
Body: [ { tx1 }, { tx2 }, { tx3 }, ... ]
```

Respuesta:
```json
{
  "total"    : 10,
  "fraudes"  : 7,
  "legitimas": 3,
  "predicciones": [...]
}
```

**Estrategia batch recomendada:** mezclad transacciones fraudulentas con transacciones legítimas en la misma llamada para simular el tráfico real. Una proporción 70% fraude / 30% legítima es realista para un ataque activo.

### Transacción "legítima" de referencia (para comparar)

Si queréis verificar que la API funciona bien, esta debería tener `prob_fraud` bajo:

```json
{
  "customer_country": "ES", "operacion_pais": "ES",
  "dispositivo_reconocido": 1,
  "numero_transacciones_ultima_hora": 1,
  "tiempo_desde_ultima_transaccion": 3600,
  "importe_transaccion": 50.00,
  "limite_importe_transacciones": 2000.00,
  "is_night": 0,
  "numero_pin_disponibles": 3,
  "destino_alto_riesgo": 0,
  "metodo_autenticacion": "PIN"
  // ... resto de campos con valores normales
}
```

---

## 8. Estrategia para Ronda 2 — después del reentrenamiento

Después de Ronda 1, el modelo se reentrena con las etiquetas reales que confirmen los analistas. El modelo habrá **aprendido exactamente vuestros patrones de Ronda 1**. Para que Ronda 2 sea un reto real:

### Qué debería cambiar en Ronda 2

**1. Evitar las señales más obvias que el modelo ya conoce:**

En lugar de combinar todos los patrones a la vez, usad ataques más sigilosos con 1-2 patrones:
- Sólo `cross_border` con `dispositivo_reconocido=1` (como si fuera un viaje)
- Importes moderados (50-60% del límite, no al 90-95%)
- `numero_transacciones_ultima_hora = 2-3` (no ráfaga obvia)

**2. Cambiar el perfil del "atacante":**
- Usar `metodo_autenticacion = "PIN"` o `"biometrico"` en lugar de `"firma"`
- `numero_pin_disponibles = 1` en lugar de 0
- `destino_alto_riesgo = 0` pero con cuenta destino nueva/inusual

**3. Aprovechar las limitaciones del modelo:**
El modelo aprendió perfiles promedio de clientes del dataset de entrenamiento. Si creáis transacciones de clientes cuyo perfil sea atípico (cliente nuevo, `tenure` bajo, `importe_medio_mensual` muy variable), las desviaciones temporales (`diff_importe_cliente`, `diff_hora_cliente`) serán imprecisas.

**4. Gradual escalation en lugar de burst instantáneo:**
En lugar de `numero_transacciones_ultima_hora = 8` de golpe, simulad un patron gradual donde cada transacción individual parece razonable pero el conjunto es un ataque.

---

## 9. Referencia rápida — valores para simular fraude

| Campo | Valor fraudulento | Valor legítimo |
|-------|------------------|---------------|
| `operacion_pais` | País diferente a `customer_country` | Igual a `customer_country` |
| `dispositivo_reconocido` | `0` | `1` |
| `numero_transacciones_ultima_hora` | `6-20` | `1-2` |
| `tiempo_desde_ultima_transaccion` | `< 55` segundos | `> 1800` segundos |
| `importe_transaccion` | `> 85%` de `limite_importe_transacciones` | `< 30%` del límite |
| `numero_pin_disponibles` | `0` | `3` |
| `metodo_autenticacion` | `"firma"` / `"3DS"` | `"PIN"` / `"biometrico"` |
| `destino_alto_riesgo` | `1` | `0` |
| `is_night` | `1` | `0` |
| `veces_superar_limite_7_dias` | `2-5` | `0` |
| `volumen_saliente_30_dias` | Mucho mayor que `vol_entrante` | Similar o menor |

---

## 10. Endpoint de referencia rápida

```
# Estado de la API
GET  http://localhost:8000/health

# Predecir una transacción
POST http://localhost:8000/predict
Body: { JSON con los campos }

# Predecir muchas transacciones
POST http://localhost:8000/predict/batch
Body: [ { tx1 }, { tx2 }, ... ]

# Ver estadísticas en tiempo real
GET  http://localhost:8000/metrics

# Documentación interactiva (podéis probar desde el navegador)
GET  http://localhost:8000/docs
```

---

*Documento generado por Data Science — Blue Team, Grupo 2 · Mayo 2026*  
*Para dudas técnicas sobre el modelo o la API, contactad con el equipo Data Science.*
