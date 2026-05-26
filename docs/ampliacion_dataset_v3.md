# Ampliación del dataset sintético — v3

**Objetivo:** llevar la tasa de fraude del 15% actual al 2-3% para que el dataset refleje proporciones realistas de producción.

---

## Problema con el dataset v2

El dataset actual (`dataset_fraude_mejorado.csv`) tiene 10.000 filas con 1.504 fraudes (15%). Esta proporción es artificialmente alta porque `calcular_probabilidad_base()` en el generador acumula muchos factores aditivos sobre una base de 0.01, haciendo que cualquier transacción con dos o tres flags de riesgo acabe siendo fraude con alta probabilidad.

El 15% de fraude en entrenamiento descalibra las probabilidades del modelo: aprende que 1 de cada 7 transacciones es fraude, cuando en producción real sería 1 de cada 33-50. Esto infla los scores de probabilidad de forma sistemática y hace que el threshold optimizado sobre este dataset no sea extrapolable.

**Nota importante:** escalar simplemente N en el generador actual no resuelve el problema. `calcular_probabilidad_base()` produce ~15% de fraude independientemente del número de transacciones generadas. Con N=50.000 seguirías obteniendo ~7.500 fraudes (15%).

---

## Estrategia de ampliación en dos capas

### Capa 1 — dataset v2 intacto

Los 10.000 registros existentes se conservan sin ninguna modificación. Los 1.504 fraudes con señal inyectada son la base de calidad del dataset y no deben tocarse.

### Capa 2 — legítimas nuevas con clientes nuevos

Generar ~40.000 transacciones legítimas puras con ~8.000 clientes nuevos (para mantener una densidad de ~5 transacciones por cliente, igual que en v2).

Reglas para la generación de esta capa:
- Misma lógica de distribución de features que el generador actual (importes, países, horarios, dispositivos, etc.)
- `IS_FRAUD = 0` fijo — no pasar por `calcular_probabilidad_base()`
- No llamar a `inyectar_senal_fraude()` bajo ningún concepto
- Usar una seed diferente a la del generador v2 (por ejemplo `seed=123`) para que los clientes y transacciones sean distintos

### Resultado combinado

| | Fraudes | Legítimas | Total | % fraude |
|---|---|---|---|---|
| Dataset v2 (existente) | 1.504 | 8.496 | 10.000 | 15.0% |
| Legítimas nuevas | 0 | ~40.000 | ~40.000 | 0% |
| **Dataset v3 combinado** | **1.504** | **~48.500** | **~50.000** | **~3.0%** |

---

## Estructura del script esperada

El script debería hacer lo siguiente:

1. Cargar el dataset v2 existente (`data/dataset_fraude_mejorado.csv`)
2. Reutilizar las funciones auxiliares del generador v2: `generar_geoloc()`, `generar_ip()`, `elegir_operacion_pais_region()`
3. Generar ~8.000 clientes nuevos con la misma lógica que el generador v2 (nivel 1)
4. Generar cuentas y tarjetas para esos clientes (niveles 2 y 3)
5. Generar ~40.000 transacciones con `IS_FRAUD = 0` fijo (sin llamar a `calcular_probabilidad_base` ni a `inyectar_senal_fraude`)
6. Añadir la columna `IMPACTO_FRAUDE` a 0 para todas las filas nuevas (igual que las legítimas del v2)
7. Concatenar dataset v2 + legítimas nuevas
8. Verificar la proporción final antes de guardar
9. Guardar en `data/dataset_fraude_v3.csv`

---

## Verificaciones antes de guardar

```python
print(f"Total filas: {len(df_v3)}")
print(f"Fraudes: {df_v3['IS_FRAUD'].sum()} ({df_v3['IS_FRAUD'].mean()*100:.2f}%)")
print(f"Columnas: {df_v3.columns.tolist()}")
assert df_v3.columns.tolist() == columnas_v2  # mismo schema exacto
assert df_v3['IS_FRAUD'].sum() == 1504        # los fraudes son solo los del v2
assert df_v3.isnull().sum().sum() == 0        # sin nulos
```

---

## Consideraciones adicionales

**Schema idéntico al v2.** El dataset v3 debe tener exactamente las mismas columnas en el mismo orden que `dataset_fraude_mejorado.csv`. El pipeline de feature engineering carga el CSV por nombre de columna, no por posición, pero mejor no arriesgar.

**No regenerar los fraudes.** Es tentador regenerar todo desde cero con una tasa de fraude menor modificando `calcular_probabilidad_base()`. No es el camino: los fraudes del v2 tienen señal inyectada manualmente y bien calibrada. Regenerar desde cero con otra lógica de probabilidad producirá fraudes con menos señal o señal distinta, rompiendo la coherencia del dataset.

**Seed reproducible.** Usar `np.random.seed(123)` y `Faker.seed(123)` para las legítimas nuevas. El v2 usa seed 42, así que hay que usar una diferente para que los clientes nuevos no colisionen en IDs.

**IDs únicos.** Verificar que no hay colisiones de `id_cliente`, `id_cuenta`, `id_tarjeta` ni `id_transaccion` entre el v2 y las legítimas nuevas. Los UUIDs truncados del generador tienen un espacio de colisión no nulo — conviene hacer `assert df_v3['id_transaccion'].nunique() == len(df_v3)` antes de guardar.
