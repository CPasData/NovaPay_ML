import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
import random
import uuid

fake = Faker('es_ES')
Faker.seed(42)
random.seed(42)
np.random.seed(42)

N = 10000

# ============================================================
# CONSTANTES Y CATEGORÍAS
# ============================================================
metodos_auth = ['PIN', 'firma', '3DS', 'huella', 'contactless']
paises_regiones = {
    'ES': ['Norte', 'Sur', 'Este', 'Oeste', 'Centro'],
    'FR': ['Île-de-France', 'Provence', 'Normandía', 'Bretaña'],
    'DE': ['Baviera', 'Berlín', 'Hamburgo', 'Hesse'],
    'IT': ['Lombardía', 'Lazio', 'Campania', 'Toscana'],
    'PT': ['Lisboa', 'Oporto', 'Algarve'],
    'GB': ['Inglaterra', 'Escocia', 'Gales'],
    'US': ['California', 'Nueva York', 'Florida', 'Texas'],
    'BR': ['São Paulo', 'Río', 'Minas Gerais'],
}
tipos_cliente = ['persona', 'empresa', 'autónomo', 'premium']
estados_cuenta = ['activa', 'bloqueada', 'suspendida', 'cerrada']
estados_tarjeta = ['activa', 'bloqueada', 'caducada', 'robada', 'extraviada']
tipos_transaccion = ['tarjeta', 'transferencia']

# ============================================================
# NIVEL 1: CLIENTES
# ============================================================
clientes = []
for _ in range(2000):
    country = np.random.choice(list(paises_regiones.keys()), p=[0.45, 0.15, 0.1, 0.08, 0.05, 0.07, 0.05, 0.05])
    clientes.append({
        'id_cliente': str(uuid.uuid4())[:12],
        'tipo_cliente': np.random.choice(tipos_cliente, p=[0.55, 0.2, 0.15, 0.1]),
        'edad_cliente': np.random.randint(18, 85),
        'customer_country': country,
        'customer_region': np.random.choice(paises_regiones[country]),
        'tenure': np.random.randint(0, 3650),
        'importe_medio_mensual': round(np.random.lognormal(6, 1.2), 2),
        'desviacion_estandar_mensual': round(np.random.exponential(500), 2),
        'media_transacciones_al_dia': round(np.random.exponential(3) + 0.5, 2),
        'numero_fraudes_ultimo_ano': 0,
    })

# ============================================================
# NIVEL 2: CUENTAS
# ============================================================
cuentas = []
for c in clientes:
    for _ in range(np.random.randint(1, 4)):
        saldo = round(np.random.lognormal(7, 1.5), 2)
        cuentas.append({
            'id_cliente': c['id_cliente'],
            'id_cuenta': str(uuid.uuid4())[:12],
            'cuenta_origen': f"ES{np.random.randint(10,99)}{np.random.randint(1000,9999)}{np.random.randint(1000,9999)}{np.random.randint(10,99)}",
            'estado_cuenta': np.random.choice(estados_cuenta, p=[0.85, 0.05, 0.05, 0.05]),
            'saldo_actual': saldo,
            'saldo_medio_30_dias': round(saldo * np.random.uniform(0.7, 1.3), 2),
            'volumen_entrante_30_dias': round(np.random.lognormal(8, 1.5), 2),
            'volumen_saliente_30_dias': round(np.random.lognormal(8, 1.5), 2),
            'numero_transferencias_recibidas_7_dias': np.random.poisson(3),
            'numero_transferencias_enviadas_7_dias': np.random.poisson(2),
        })

# ============================================================
# NIVEL 3: TARJETAS
# ============================================================
tarjetas = []
for cu in cuentas:
    for _ in range(np.random.randint(1, 3)):
        fecha_creacion = fake.date_between(start_date='-5y', end_date='today')
        tarjetas.append({
            'id_cuenta': cu['id_cuenta'],
            'id_cliente': cu['id_cliente'],
            'id_tarjeta': str(uuid.uuid4())[:12],
            'estado_tarjeta': np.random.choice(estados_tarjeta, p=[0.8, 0.05, 0.08, 0.04, 0.03]),
            'fecha_creacion_tarjeta': fecha_creacion,
            'antiguedad_tarjeta_dias': (datetime.now().date() - fecha_creacion).days,
            'limite_importe_transacciones': round(np.random.choice([500, 1000, 2000, 3000, 5000], p=[0.2, 0.3, 0.3, 0.15, 0.05]), 2),
            'veces_superar_limite_7_dias': np.random.poisson(0.3),
        })

# ============================================================
# NIVEL EXTRA: CUENTAS DESTINO (algunas marcadas como fraude)
# ============================================================
cuentas_destino = []
for _ in range(500):
    pais = np.random.choice(list(paises_regiones.keys()))
    es_fraude = 1 if np.random.random() < 0.12 else 0
    cuentas_destino.append({
        'cuenta_destino': f"ES{np.random.randint(10,99)}{np.random.randint(1000,9999)}{np.random.randint(1000,9999)}{np.random.randint(10,99)}",
        'destino_alto_riesgo': es_fraude,
    })

# ============================================================
# FUNCIONES AUXILIARES
# ============================================================
def generar_geoloc(country):
    ciudades = {
        'ES': ('40.4168,-3.7038', '41.3874,2.1686', '37.3891,-5.9845', '39.4699,-0.3763'),
        'FR': ('48.8566,2.3522', '45.7640,4.8357', '43.2965,5.3698'),
        'DE': ('52.5200,13.4050', '48.1351,11.5820', '53.5511,9.9937'),
        'IT': ('41.9028,12.4964', '45.4642,9.1900', '40.8518,14.2681'),
        'PT': ('38.7223,-9.1393', '41.1579,-8.6291'),
        'GB': ('51.5074,-0.1278', '55.9533,-3.1883', '53.4808,-2.2426'),
        'US': ('34.0522,-118.2437', '40.7128,-74.0060', '25.7617,-80.1918', '29.7604,-95.3698'),
        'BR': ('-23.5505,-46.6333', '-22.9068,-43.1729', '-19.9167,-43.9345'),
    }
    return np.random.choice(ciudades.get(country, ('40.4168,-3.7038',)))

def generar_ip(country):
    first = {'ES': 86, 'FR': 90, 'DE': 87, 'IT': 95, 'PT': 85, 'GB': 81, 'US': 73, 'BR': 177}
    f = first.get(country, 86)
    return f"{f}.{np.random.randint(0,256)}.{np.random.randint(0,256)}.{np.random.randint(1,255)}"

def elegir_operacion_pais_region(home_country, home_region):
    """
    El 75% de las veces la operación es en el mismo país y región del cliente.
    El 15% mismo país, región diferente.
    El 10% país diferente.
    """
    r = np.random.random()
    if r < 0.75:
        return home_country, home_region
    elif r < 0.90:
        regiones = [r for r in paises_regiones[home_country] if r != home_region]
        region = np.random.choice(regiones) if regiones else home_region
        return home_country, region
    else:
        paises = [p for p in paises_regiones if p != home_country]
        pais = np.random.choice(paises)
        region = np.random.choice(paises_regiones[pais])
        return pais, region

def calcular_probabilidad_fraude(row):
    """
    PATRÓN DE FRAUDE - SUMA DE PORCENTAJES
    =======================================
    Base:                        1%
    + País operación != home:   15%
    + Región operación != home:  5%  (solo si mismo país)
    + Dispositivo no reconocido:10%
    + Cuenta bloqueada:         15%
    + Tarjeta robada/extraviada:25%
    + Importe > 90% límite:     10%
    + Noche + importe > 500:     5%
    + Vol. saliente > 3x entrante:3%
    + >5 transacciones última h: 8%
    + Última transac <60s y >1000:6%
    + Veces superar límite >3:  10%
    + Método autenticación firma:2%
    + PIN disponibles = 0:       8%
    + Tipo transferencia:        5%
    + Cuenta destino alto riesgo:20%
    Cap máximo:                 95%
    """
    prob = 0.01

    if row['customer_country'] != row['operacion_pais']:
        prob += 0.15
    elif row['customer_region'] != row['operacion_region']:
        prob += 0.05

    if row['dispositivo_reconocido'] == 0:
        prob += 0.10

    if row['estado_cuenta'] == 'bloqueada':
        prob += 0.15

    if row['estado_tarjeta'] in ('robada', 'extraviada', 'bloqueada'):
        prob += 0.25

    if row['importe_transaccion'] > row['limite_importe_transacciones'] * 0.9:
        prob += 0.10

    if row['is_night'] and row['importe_transaccion'] > 500:
        prob += 0.05

    if row['volumen_saliente_30_dias'] > row['volumen_entrante_30_dias'] * 3:
        prob += 0.03

    if row['numero_transacciones_ultima_hora'] > 5:
        prob += 0.08

    if row['tiempo_desde_ultima_transaccion'] < 60 and row['importe_transaccion'] > 1000:
        prob += 0.06

    if row['veces_superar_limite_7_dias'] > 3:
        prob += 0.10

    if row['metodo_autenticacion'] == 'firma':
        prob += 0.02

    if row['numero_pin_disponibles'] == 0:
        prob += 0.08

    if row['tipo_transaccion'] == 'transferencia':
        prob += 0.05

    if row.get('destino_alto_riesgo', 0) == 1:
        prob += 0.20

    return min(prob, 0.95)

# ============================================================
# NIVEL 4: TRANSACCIONES
# ============================================================
registros = []
ultima_transaccion_por_tarjeta = {}
ids_cuenta_destino_usados = []

for _ in range(N):
    t = np.random.choice(tarjetas)
    cu = next(c for c in cuentas if c['id_cuenta'] == t['id_cuenta'])
    c = next(cl for cl in clientes if cl['id_cliente'] == t['id_cliente'])
    dest = np.random.choice(cuentas_destino)

    fecha_hora = fake.date_time_between(start_date='-6M', end_date='now')
    is_night = 1 if fecha_hora.hour < 6 or fecha_hora.hour >= 23 else 0
    is_weekend = 1 if fecha_hora.weekday() >= 5 else 0

    ult = ultima_transaccion_por_tarjeta.get(t['id_tarjeta'])
    if ult:
        diff = (fecha_hora - ult).total_seconds()
        tiempo_desde_ultima = max(0, int(diff))
    else:
        tiempo_desde_ultima = np.random.randint(3600, 86400)
    ultima_transaccion_por_tarjeta[t['id_tarjeta']] = fecha_hora

    importe = round(np.random.lognormal(5, 1.5), 2)
    operacion_pais, operacion_region = elegir_operacion_pais_region(c['customer_country'], c['customer_region'])
    tipo_tx = np.random.choice(tipos_transaccion, p=[0.7, 0.3])

    row = {
        # Cliente
        'id_cliente': c['id_cliente'],
        'tipo_cliente': c['tipo_cliente'],
        'edad_cliente': c['edad_cliente'],
        'customer_country': c['customer_country'],
        'customer_region': c['customer_region'],
        'tenure': c['tenure'],
        'importe_medio_mensual': c['importe_medio_mensual'],
        'desviacion_estandar_mensual': c['desviacion_estandar_mensual'],
        'media_transacciones_al_dia': c['media_transacciones_al_dia'],
        'numero_fraudes_ultimo_ano': c['numero_fraudes_ultimo_ano'],
        # Cuenta origen
        'id_cuenta': cu['id_cuenta'],
        'cuenta_origen': cu['cuenta_origen'],
        'estado_cuenta': cu['estado_cuenta'],
        'saldo_actual': cu['saldo_actual'],
        'saldo_medio_30_dias': cu['saldo_medio_30_dias'],
        'volumen_entrante_30_dias': cu['volumen_entrante_30_dias'],
        'volumen_saliente_30_dias': cu['volumen_saliente_30_dias'],
        'numero_transferencias_recibidas_7_dias': cu['numero_transferencias_recibidas_7_dias'],
        'numero_transferencias_enviadas_7_dias': cu['numero_transferencias_enviadas_7_dias'],
        # Tarjeta
        'id_tarjeta': t['id_tarjeta'],
        'estado_tarjeta': t['estado_tarjeta'],
        'fecha_creacion_tarjeta': t['fecha_creacion_tarjeta'],
        'antiguedad_tarjeta_dias': t['antiguedad_tarjeta_dias'],
        'limite_importe_transacciones': t['limite_importe_transacciones'],
        'veces_superar_limite_7_dias': t['veces_superar_limite_7_dias'],
        # Transacción
        'id_transaccion': str(uuid.uuid4())[:12],
        'tipo_transaccion': tipo_tx,
        'fecha_hora': fecha_hora,
        'is_night': is_night,
        'is_weekend': is_weekend,
        'tiempo_desde_ultima_transaccion': tiempo_desde_ultima,
        'numero_transacciones_ultima_hora': np.random.poisson(2),
        'importe_transaccion': importe,
        'metodo_autenticacion': np.random.choice(metodos_auth, p=[0.4, 0.15, 0.3, 0.1, 0.05]),
        'numero_pin_disponibles': np.random.choice([0, 1, 2, 3], p=[0.02, 0.30, 0.50, 0.18]),
        # Dispositivo y ubicación
        'identificador_dispositivo_fingerprint': str(uuid.uuid4())[:16],
        'dispositivo_reconocido': 1 if np.random.random() < 0.85 else 0,
        'operacion_pais': operacion_pais,
        'operacion_region': operacion_region,
        'direccion_ip_origen': generar_ip(operacion_pais),
        'geolocalizacion': generar_geoloc(operacion_pais),
        # Cuenta destino
        'cuenta_destino': dest['cuenta_destino'],
        'destino_alto_riesgo': dest['destino_alto_riesgo'],
        # Etiqueta
        'IS_FRAUD': 0,
    }

    prob_fraude = calcular_probabilidad_fraude(row)
    row['IS_FRAUD'] = 1 if np.random.random() < prob_fraude else 0

    if row['IS_FRAUD'] == 1:
        c['numero_fraudes_ultimo_ano'] += 1

    registros.append(row)

# ============================================================
# EXPORTACIÓN
# ============================================================
df = pd.DataFrame(registros)
print(f"Total registros: {len(df)}")
print(f"Fraudes: {df['IS_FRAUD'].sum()} ({df['IS_FRAUD'].mean()*100:.2f}%)")

columnas_final = [
    'id_cliente', 'tipo_cliente', 'edad_cliente',
    'customer_country', 'customer_region', 'tenure',
    'importe_medio_mensual', 'desviacion_estandar_mensual',
    'media_transacciones_al_dia', 'numero_fraudes_ultimo_ano',
    'id_cuenta', 'cuenta_origen', 'estado_cuenta',
    'saldo_actual', 'saldo_medio_30_dias',
    'volumen_entrante_30_dias', 'volumen_saliente_30_dias',
    'numero_transferencias_recibidas_7_dias', 'numero_transferencias_enviadas_7_dias',
    'id_tarjeta', 'estado_tarjeta', 'fecha_creacion_tarjeta',
    'antiguedad_tarjeta_dias', 'limite_importe_transacciones',
    'veces_superar_limite_7_dias',
    'id_transaccion', 'tipo_transaccion', 'fecha_hora',
    'is_night', 'is_weekend',
    'tiempo_desde_ultima_transaccion', 'numero_transacciones_ultima_hora',
    'importe_transaccion', 'metodo_autenticacion', 'numero_pin_disponibles',
    'identificador_dispositivo_fingerprint', 'dispositivo_reconocido',
    'operacion_pais', 'operacion_region',
    'direccion_ip_origen', 'geolocalizacion',
    'cuenta_destino', 'destino_alto_riesgo',
    'IS_FRAUD',
]

df = df[columnas_final]
df.to_csv('dataset_fraude.csv', index=False, encoding='utf-8')
print("Dataset guardado en dataset_fraude.csv")
