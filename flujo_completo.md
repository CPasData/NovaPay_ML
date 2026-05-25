# Flujo completo del sistema NovaPay — Operación Centinela

---

## Ronda 1 — Flujo de una transacción

```
ENTRA UNA TRANSACCIÓN AL SISTEMA
        ↓
┌─────────────────────────────────────────┐
│  FULL STACK                             │
│  llama a nuestra API:                   │
│  POST /predict                          │
│  con los datos de la transacción        │
└─────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────┐
│  DATA SCIENCE — API FastAPI             │
│  1. Aplica el modelo ML                 │
│  2. Calcula is_fraud + prob_fraud       │
│  3. Guarda TODO en BD                   │
│     → is_fraud, prob_fraud              │
│     → impacto_fraude                    │
│     → campos calculados del modelo      │
│     → estado_revision = 'pendiente'     │
│  4. Devuelve resultado a Full Stack     │
└─────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────┐
│  FULL STACK                             │
│  Recibe:                                │
│  { is_fraud: 1, prob_fraud: 0.87 }      │
│  Muestra en la app del analista         │
└─────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────┐
│  ANALISTA                               │
│  Ve la lista de transacciones           │
│  sospechosas ordenadas por prob_fraud   │
│  Abre cada caso y decide:               │
└─────────────────────────────────────────┘
        ↓
   ┌────┴────┐
   ↓         ↓
¿ES FRAUDE?  ¿NO ES FRAUDE?
"Confirmar"  "Falso positivo"
   ↓         ↓
   └────┬────┘
        ↓
┌─────────────────────────────────────────┐
│  FULL STACK                             │
│  Actualiza en Base de datos:            │
│  → target_final = TRUE/FALSE            │
│  → estado_revision = confirmado_fraude  │
│                    / falso_positivo     │
│  → id_usuario = analista que revisó     │
│  → fecha_revision = ahora               │
└─────────────────────────────────────────┘
```

---

## ¿Qué pasa después según la decisión del analista?

### Si el analista confirma fraude ✅
```
target_final    = TRUE
estado_revision = 'confirmado_fraude'
        ↓
Full Stack guarda en Supabase
        ↓
Data Science en Ronda 2:
"Data Science tiene ahora etiquetas REALES
no inventadas sino confirmadas por un humano"
→ lee los casos con target_final = TRUE
→ los usa para reentrenar el modelo
→ "aprende" cómo era ese fraude real
→ despliega modelo mejorado
```

### Si el analista dice que NO es fraude ❌
```
target_final    = FALSE
estado_revision = 'falso_positivo'
        ↓
Full Stack guarda en Supabase
        ↓
Data Science en Ronda 2:
→ lee los casos con target_final = FALSE
→ los usa para reentrenar el modelo
→ "aprende" que ese patrón NO es fraude
→ el modelo deja de marcar ese patrón
→ menos falsos positivos en Ronda 2
```

---
## Fin de Ronda 1 y preparación para Ronda 2

```
RONDA 1 TERMINADA ✅
        ↓
El analista termina de revisar TODOS los casos pendientes
→ no quedan transacciones con estado_revision = 'pendiente'
        ↓
Full Stack habilita el botón:
"Reentrenar modelo"
→ el botón solo aparece cuando
  pendientes_revision = 0 en /metrics
        ↓
Full Stack llama a:
POST /retrain
        ↓
Data Science — API hace automáticamente:
1. Lee desde la BD de y consulta todas las transacciones
que el analista reviso:
   SELECT * FROM transacciones
   WHERE target_final IS NOT NULL
2. Reentrena el modelo con target_final
   como etiqueta real (no IS_FRAUD del CSV)
3. Genera modelo_ronda2.pkl
4. Reemplaza el pkl en la API
5. Reinicia con el modelo mejorado
        ↓
API responde:
{
  "estado"           : "modelo actualizado ✅",
  "precision_ronda1" : "65%",
  "precision_ronda2" : "82%"
}
        ↓
RONDA 2 COMIENZA 
Ciber vuelve a atacar con fraude más sigiloso
El modelo mejorado detecta más fraudes
```
---

## Resumen de responsabilidades

| Tarea | Equipo |
|-------|--------|
| Llamar a POST /predict | Full Stack |
| Predecir con el modelo | Data Science |
| Guardar predicción en Base de datos | Data Science |
| Mostrar casos al analista | Full Stack |
| Actualizar target_final y estado_revision | Full Stack |
| Mostrar botón "Reentrenar" cuando pendientes = 0 | Full Stack |
| Llamar a POST /retrain cuando analista termina | Full Stack |
| Leer datos confirmados y reentrenar el modelo | Data Science |
| Reentrenar y activar modelo mejorado automáticamente | Data Science |
| Atacar la API con transacciones sospechosas | Ciberseguridad |

---

## Conexión a Supabase

Cuando Full Stack tenga Supabase lista, Data Science solo cambia
una línea en `app.py`:

```python
# Ahora (local):
DB_CONFIG = {
    "host": "localhost",
    ...
}

# Cuando Supabase esté lista:
DB_CONFIG = {
    "host"    : "db.xxxxxxxxxxxx.supabase.co",
    "port"    : 5432,
    "database": "postgres",
    "user"    : "postgres",
    "password": "contraseña_supabase"
}
```

El resto del código no cambia. ✅
