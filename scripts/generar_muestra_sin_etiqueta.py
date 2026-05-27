"""
Genera archivos JSON y CSV de transacciones sintéticas SIN etiqueta (IS_FRAUD).
Por defecto genera una muestra mixta. Con --perfil genera variantes especializadas.

Útil para:
  - Simular producción (datos sin ground truth)
  - Probar la API de inferencia
  - Evaluar comportamiento del modelo en distintos perfiles de riesgo

Uso:
    python scripts/generar_muestra_sin_etiqueta.py
    # Genera data/muestra_sin_etiqueta.csv + .json (mixto, 200 tx)

    python scripts/generar_muestra_sin_etiqueta.py --perfil fraude
    # Genera data/muestra_fraude.csv + .json (200 tx con perfil fraudulento)

    python scripts/generar_muestra_sin_etiqueta.py --perfil todo
    # Genera los 3 perfiles + mixto (4 pares CSV+JSON)
"""

import argparse
import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
from pathlib import Path
from uuid import uuid4

fake = Faker('es_ES')
Faker.seed(42)

PAISES_REGIONES = {
    'ES': ['Norte', 'Sur', 'Este', 'Oeste', 'Centro'],
    'FR': ['Île-de-France', 'Provence', 'Normandía', 'Bretaña'],
    'DE': ['Baviera', 'Berlín', 'Hamburgo', 'Hesse'],
    'IT': ['Lombardía', 'Lazio', 'Campania', 'Toscana'],
    'PT': ['Lisboa', 'Oporto', 'Algarve'],
    'GB': ['Inglaterra', 'Escocia', 'Gales'],
    'US': ['California', 'Nueva York', 'Florida', 'Texas'],
    'BR': ['São Paulo', 'Río', 'Minas Gerais'],
}
TIPOS_CLIENTE = ['persona', 'empresa', 'autónomo', 'premium']
ESTADOS_CUENTA = ['activa', 'bloqueada', 'suspendida', 'cerrada']
ESTADOS_TARJETA = ['activa', 'bloqueada', 'caducada', 'robada', 'extraviada']
TIPOS_TRANSACCION = ['tarjeta', 'transferencia', 'bizum']
METODOS_AUTH = ['PIN', 'firma', '3DS', 'huella', 'contactless']

COLUMNAS = [
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
]


PERFILES = {
    'mixto': {
        'label': 'Muestra mixta (patrón estándar)',
        'p_dispositivo_reconocido': 0.85,
        'p_cross_border': 0.10,
        'p_destino_alto_riesgo': 0.12,
        'p_estados_tarjeta': [0.80, 0.05, 0.08, 0.04, 0.03],
        'importe_log_mean': 5,
        'poisson_tx_hora': 2,
        'p_night': 0.15,
    },
    'sospechoso': {
        'label': 'Perfil sospechoso (señales de riesgo elevadas)',
        'p_dispositivo_reconocido': 0.50,
        'p_cross_border': 0.40,
        'p_destino_alto_riesgo': 0.35,
        'p_estados_tarjeta': [0.60, 0.05, 0.10, 0.15, 0.10],
        'importe_log_mean': 6,
        'poisson_tx_hora': 5,
        'p_night': 0.30,
    },
    'fraude': {
        'label': 'Perfil fraudulento (señales de fraude muy marcadas)',
        'p_dispositivo_reconocido': 0.20,
        'p_cross_border': 0.70,
        'p_destino_alto_riesgo': 0.60,
        'p_estados_tarjeta': [0.30, 0.05, 0.10, 0.35, 0.20],
        'importe_log_mean': 7,
        'poisson_tx_hora': 10,
        'p_night': 0.40,
    },
}


def generar_muestra(n=200, seed=42, perfil='mixto'):
    cfg = PERFILES.get(perfil, PERFILES['mixto'])
    np.random.seed(seed)
    filas = []

    for _ in range(n):
        country = np.random.choice(list(PAISES_REGIONES.keys()),
                                   p=[0.45, 0.15, 0.1, 0.08, 0.05, 0.07, 0.05, 0.05])

        cliente = {
            'id_cliente': str(uuid4())[:12],
            'tipo_cliente': np.random.choice(TIPOS_CLIENTE, p=[0.55, 0.2, 0.15, 0.1]),
            'edad_cliente': int(np.random.randint(18, 85)),
            'customer_country': country,
            'customer_region': np.random.choice(PAISES_REGIONES[country]),
            'tenure': int(np.random.randint(0, 3650)),
            'importe_medio_mensual': round(float(np.random.lognormal(6, 1.2)), 2),
            'desviacion_estandar_mensual': round(float(np.random.exponential(500)), 2),
            'media_transacciones_al_dia': round(float(np.random.exponential(3) + 0.5), 2),
            'numero_fraudes_ultimo_ano': int(0),
        }

        saldo = round(float(np.random.lognormal(7, 1.5)), 2)
        cuenta = {
            'id_cuenta': str(uuid4())[:12],
            'cuenta_origen': f"ES{np.random.randint(10,99)}{np.random.randint(1000,9999)}"
                             f"{np.random.randint(1000,9999)}{np.random.randint(10,99)}",
            'estado_cuenta': np.random.choice(ESTADOS_CUENTA, p=[0.85, 0.05, 0.05, 0.05]),
            'saldo_actual': saldo,
            'saldo_medio_30_dias': round(saldo * np.random.uniform(0.7, 1.3), 2),
            'volumen_entrante_30_dias': round(float(np.random.lognormal(8, 1.5)), 2),
            'volumen_saliente_30_dias': round(float(np.random.lognormal(8, 1.5)), 2),
            'numero_transferencias_recibidas_7_dias': int(np.random.poisson(3)),
            'numero_transferencias_enviadas_7_dias': int(np.random.poisson(2)),
        }

        p_tarjeta = cfg['p_estados_tarjeta']

        fecha_creacion = fake.date_between(start_date='-5y', end_date='today')
        tarjeta = {
            'id_tarjeta': str(uuid4())[:12],
            'estado_tarjeta': np.random.choice(ESTADOS_TARJETA, p=p_tarjeta),
            'fecha_creacion_tarjeta': fecha_creacion,
            'antiguedad_tarjeta_dias': (datetime.now().date() - fecha_creacion).days,
            'limite_importe_transacciones': round(float(np.random.choice(
                [500, 1000, 2000, 3000, 5000], p=[0.2, 0.3, 0.3, 0.15, 0.05])), 2),
            'veces_superar_limite_7_dias': int(np.random.poisson(
                0.3 if perfil == 'mixto' else 2)),
        }

        destino = {
            'cuenta_destino': f"ES{np.random.randint(10,99)}{np.random.randint(1000,9999)}"
                              f"{np.random.randint(1000,9999)}{np.random.randint(10,99)}",
            'destino_alto_riesgo': int(1 if np.random.random() < cfg['p_destino_alto_riesgo'] else 0),
        }

        fecha_hora = fake.date_time_between(start_date='-6M', end_date='now')
        is_night = 1 if (fecha_hora.hour < 6 or fecha_hora.hour >= 23) or np.random.random() < cfg['p_night'] else 0
        is_weekend = 1 if fecha_hora.weekday() >= 5 else 0

        r = np.random.random()
        if r < 1 - cfg['p_cross_border']:
            op_pais, op_region = cliente['customer_country'], cliente['customer_region']
        elif r < 1 - cfg['p_cross_border'] * 0.5:
            regiones = [r for r in PAISES_REGIONES[cliente['customer_country']]
                        if r != cliente['customer_region']]
            op_region = np.random.choice(regiones) if regiones else cliente['customer_region']
            op_pais = cliente['customer_country']
        else:
            otros = [p for p in PAISES_REGIONES if p != cliente['customer_country']]
            op_pais = np.random.choice(otros)
            op_region = np.random.choice(PAISES_REGIONES[op_pais])

        # Ajustar hora según perfil: más fraudes ocurren de noche
        if is_night and perfil in ('sospechoso', 'fraude'):
            hora_base = np.random.randint(0, 5) if np.random.random() < 0.6 else np.random.randint(23, 24)
            fecha_hora = fecha_hora.replace(hour=hora_base, minute=np.random.randint(0, 59))

        tx = {
            'id_transaccion': str(uuid4())[:12],
            'tipo_transaccion': np.random.choice(TIPOS_TRANSACCION, p=[0.6, 0.25, 0.15]),
            'fecha_hora': fecha_hora.isoformat(),
            'is_night': int(is_night),
            'is_weekend': int(is_weekend),
            'tiempo_desde_ultima_transaccion': int(np.random.randint(
                5 if perfil == 'fraude' else 30, 86400)),
            'numero_transacciones_ultima_hora': int(np.random.poisson(cfg['poisson_tx_hora'])),
            'importe_transaccion': round(float(np.random.lognormal(cfg['importe_log_mean'], 1.5)), 2),
            'metodo_autenticacion': np.random.choice(METODOS_AUTH, p=[0.4, 0.15, 0.3, 0.1, 0.05]),
            'numero_pin_disponibles': int(np.random.choice(
                [0, 1, 2, 3],
                p=[0.02 if perfil == 'mixto' else 0.15, 0.30, 0.50, 0.18 - (0.13 if perfil != 'mixto' else 0)]
            )),
            'identificador_dispositivo_fingerprint': str(uuid4())[:16],
            'dispositivo_reconocido': int(1 if np.random.random() < cfg['p_dispositivo_reconocido'] else 0),
            'operacion_pais': op_pais,
            'operacion_region': op_region,
            'direccion_ip_origen': f"{86}.{np.random.randint(0,256)}.{np.random.randint(0,256)}.{np.random.randint(1,255)}",
            'geolocalizacion': "40.4168,-3.7038",
        }

        filas.append({**cliente, **cuenta, **tarjeta, **destino, **tx})

    df = pd.DataFrame(filas)
    return df[COLUMNAS]


def _guardar(df, base_path):
    base_path.parent.mkdir(exist_ok=True)
    csv_path = base_path.with_suffix('.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8')
    json_path = base_path.with_suffix('.json')
    df.to_json(json_path, orient='records', indent=2, force_ascii=False, date_format='iso')
    print(f"  CSV:  {csv_path}")
    print(f"  JSON: {json_path}")
    return csv_path, json_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generar CSV/JSON de transacciones sin etiqueta (patrón v3)')
    parser.add_argument('--n', type=int, default=200, help='Número de transacciones')
    parser.add_argument('--seed', type=int, default=42, help='Semilla aleatoria')
    parser.add_argument('--output', type=str, default='data/muestra_sin_etiqueta',
                        help='Ruta base de salida (sin extensión) para perfil individual')
    parser.add_argument('--perfil', type=str, default='mixto',
                        choices=['mixto', 'sospechoso', 'fraude', 'todo'],
                        help='Perfil de riesgo de las transacciones')
    args = parser.parse_args()

    perfiles_a_generar = PERFILES.keys() if args.perfil == 'todo' else [args.perfil]

    total_generados = 0
    for perfil in perfiles_a_generar:
        cfg = PERFILES[perfil]
        if args.perfil == 'todo':
            base_path = Path(__file__).resolve().parent.parent / f'data/muestra_{perfil}'
        else:
            base_path = Path(__file__).resolve().parent.parent / args.output

        print(f"\nGenerando: {cfg['label']} ({args.n} tx)")
        df = generar_muestra(n=args.n, seed=args.seed, perfil=perfil)
        csv_p, json_p = _guardar(df, base_path)
        print(f"  Columnas: {len(df.columns)}")
        total_generados += len(df)

    print(f"\nTotal: {total_generados} transacciones generadas")
    if args.perfil == 'todo':
        print("\nArchivos generados:")
        for p in PERFILES:
            print(f"  data/muestra_{p}.csv  +  data/muestra_{p}.json"
                  f"  ({PERFILES[p]['label']})")
