"""
Evaluación de modelos de fraude por rondas de 100 transacciones sintéticas.

Genera datos en lotes secuenciales, ejecuta inferencia, y acumula métricas
por ronda para simular producción y detectar degradación.

Uso:
    python scripts/evaluacion_rondas.py
    python scripts/evaluacion_rondas.py --modelo v2 --rondas 50 --drift suave --output resultados.csv

Escenarios de drift:
    - baseline:   sin drift (distribución estable)
    - suave:      cambio gradual en features de fraude (ronda 20-40)
    - abrupto:    cambio brusco en ronda 30
    - concepto:   la relación features → fraude cambia (nuevos patrones no vistos)
"""

import sys, joblib, json, warnings, argparse
warnings.filterwarnings('ignore')
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from faker import Faker
from uuid import uuid4
from sklearn.metrics import (
    precision_score, recall_score, fbeta_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, roc_curve
)
from scipy.stats import ks_2samp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_engineering import FeatureEngineer


# CONSTANTES (mismas que los generadores)

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

TOP_LEVEL_KEYS = {
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
}


def generar_ronda(n=100, seed=None, drift_cfg=None, ronda_idx=0):
    """Genera una ronda de n transacciones sintéticas.

    drift_cfg controla desvíos progresivos respecto a la distribución base:
        None o {} → baseline (distribución estable)
        {'tipo': 'suave', 'inicio': 20, 'fin': 40, 'feature': 'dispositivo_reconocido', 'magnitud': 0.3}
        {'tipo': 'abrupto', 'ronda': 30, 'feature': 'operacion_pais', 'swap_prob': 0.5}
        {'tipo': 'concepto', 'ronda': 25, 'nuevo_patron': 'importe_bajo_madrugada'}
    """
    if seed is not None:
        np.random.seed(seed)
        random_state = np.random.RandomState(seed)
    else:
        random_state = np.random.RandomState()

    filas = []
    for _ in range(n):
        country = random_state.choice(list(PAISES_REGIONES.keys()),
                                      p=[0.45, 0.15, 0.1, 0.08, 0.05, 0.07, 0.05, 0.05])

        cliente = {
            'id_cliente': str(uuid4())[:12],
            'tipo_cliente': random_state.choice(TIPOS_CLIENTE, p=[0.55, 0.2, 0.15, 0.1]),
            'edad_cliente': int(random_state.randint(18, 85)),
            'customer_country': country,
            'customer_region': random_state.choice(PAISES_REGIONES[country]),
            'tenure': int(random_state.randint(0, 3650)),
            'importe_medio_mensual': round(float(random_state.lognormal(6, 1.2)), 2),
            'desviacion_estandar_mensual': round(float(random_state.exponential(500)), 2),
            'media_transacciones_al_dia': round(float(random_state.exponential(3) + 0.5), 2),
            'numero_fraudes_ultimo_ano': 0,
        }

        saldo = round(float(random_state.lognormal(7, 1.5)), 2)
        cuenta = {
            'id_cuenta': str(uuid4())[:12],
            'cuenta_origen': f"ES{random_state.randint(10,99)}{random_state.randint(1000,9999)}"
                             f"{random_state.randint(1000,9999)}{random_state.randint(10,99)}",
            'estado_cuenta': random_state.choice(ESTADOS_CUENTA, p=[0.85, 0.05, 0.05, 0.05]),
            'saldo_actual': saldo,
            'saldo_medio_30_dias': round(saldo * random_state.uniform(0.7, 1.3), 2),
            'volumen_entrante_30_dias': round(float(random_state.lognormal(8, 1.5)), 2),
            'volumen_saliente_30_dias': round(float(random_state.lognormal(8, 1.5)), 2),
            'numero_transferencias_recibidas_7_dias': int(random_state.poisson(3)),
            'numero_transferencias_enviadas_7_dias': int(random_state.poisson(2)),
        }

        fecha_creacion = fake.date_between(start_date='-5y', end_date='today')
        tarjeta = {
            'id_tarjeta': str(uuid4())[:12],
            'estado_tarjeta': random_state.choice(ESTADOS_TARJETA, p=[0.8, 0.05, 0.08, 0.04, 0.03]),
            'fecha_creacion_tarjeta': fecha_creacion,
            'antiguedad_tarjeta_dias': (datetime.now().date() - fecha_creacion).days,
            'limite_importe_transacciones': round(float(random_state.choice(
                [500, 1000, 2000, 3000, 5000], p=[0.2, 0.3, 0.3, 0.15, 0.05])), 2),
            'veces_superar_limite_7_dias': int(random_state.poisson(0.3)),
        }

        dest_pais = random_state.choice(list(PAISES_REGIONES.keys()))
        destino = {
            'cuenta_destino': f"ES{random_state.randint(10,99)}{random_state.randint(1000,9999)}"
                              f"{random_state.randint(1000,9999)}{random_state.randint(10,99)}",
            'destino_alto_riesgo': 1 if random_state.random() < 0.12 else 0,
        }

        fecha_hora = fake.date_time_between(start_date='-6M', end_date='now')
        is_night = 1 if fecha_hora.hour < 6 or fecha_hora.hour >= 23 else 0
        is_weekend = 1 if fecha_hora.weekday() >= 5 else 0
        importe = round(float(random_state.lognormal(5, 1.5)), 2)

        r = random_state.random()
        if r < 0.75:
            op_pais, op_region = cliente['customer_country'], cliente['customer_region']
        elif r < 0.90:
            regiones = [r for r in PAISES_REGIONES[cliente['customer_country']]
                        if r != cliente['customer_region']]
            op_region = random_state.choice(regiones) if regiones else cliente['customer_region']
            op_pais = cliente['customer_country']
        else:
            otros = [p for p in PAISES_REGIONES if p != cliente['customer_country']]
            op_pais = random_state.choice(otros)
            op_region = random_state.choice(PAISES_REGIONES[op_pais])

        tipo_tx = random_state.choice(TIPOS_TRANSACCION, p=[0.6, 0.25, 0.15])

        row = {**cliente, **cuenta, **tarjeta, **destino,
               'id_transaccion': str(uuid4())[:12],
               'tipo_transaccion': tipo_tx,
               'fecha_hora': fecha_hora.isoformat(),
               'is_night': is_night,
               'is_weekend': is_weekend,
               'tiempo_desde_ultima_transaccion': int(random_state.randint(30, 86400)),
               'numero_transacciones_ultima_hora': int(random_state.poisson(2)),
               'importe_transaccion': importe,
               'metodo_autenticacion': random_state.choice(METODOS_AUTH, p=[0.4, 0.15, 0.3, 0.1, 0.05]),
               'numero_pin_disponibles': int(random_state.choice([0, 1, 2, 3], p=[0.02, 0.30, 0.50, 0.18])),
               'identificador_dispositivo_fingerprint': str(uuid4())[:16],
               'dispositivo_reconocido': 1 if random_state.random() < 0.85 else 0,
               'operacion_pais': op_pais,
               'operacion_region': op_region,
               'direccion_ip_origen': f"{86}.{random_state.randint(0,256)}.{random_state.randint(0,256)}.{random_state.randint(1,255)}",
               'geolocalizacion': "40.4168,-3.7038",
               }
        row['IS_FRAUD'] = 0

        
        # ASIGNAR ETIQUETA (con o sin drift)
        
        prob = _calc_base_prob(row)

        # Concept drift: cambiar la relación features → fraude
        if drift_cfg and drift_cfg.get('tipo') == 'concepto' and ronda_idx >= drift_cfg.get('ronda', 25):
            if drift_cfg.get('nuevo_patron') == 'importe_bajo_madrugada':
                if row['is_night'] and row['importe_transaccion'] < 100:
                    prob = min(prob + 0.4, 0.95)
            elif drift_cfg.get('nuevo_patron') == 'tarjeta_activa_fraude':
                if row['estado_tarjeta'] == 'activa' and row['tipo_transaccion'] == 'transferencia':
                    prob = min(prob + 0.35, 0.95)

        row['IS_FRAUD'] = 1 if random_state.random() < prob else 0

        
        # INYECTAR SEÑAL (con drift en features)
        
        if row['IS_FRAUD'] == 1:
            row = _inyectar_senal(row, random_state)
            # Drift suave/abrupto: modificar cómo se inyecta la señal
            if drift_cfg and drift_cfg.get('tipo') in ('suave', 'abrupto'):
                row = _aplicar_drift_feature(row, drift_cfg, ronda_idx, random_state)

        filas.append(row)

    df = pd.DataFrame(filas)
    df = df[[c for c in TOP_LEVEL_KEYS if c in df.columns]]
    return df


def _calc_base_prob(row):
    prob = 0.005
    cm = row['customer_country'] != row['operacion_pais']
    rm = (row['customer_country'] == row['operacion_pais'] and
          row['customer_region'] != row['operacion_region'])
    if cm: prob += 0.03
    elif rm: prob += 0.01
    if row['dispositivo_reconocido'] == 0: prob += 0.02
    if row['estado_cuenta'] == 'bloqueada': prob += 0.03
    if row['estado_tarjeta'] in ('robada', 'extraviada', 'bloqueada'): prob += 0.05
    if row['importe_transaccion'] > row['limite_importe_transacciones'] * 0.9: prob += 0.02
    if row['is_night'] and row['importe_transaccion'] > 500: prob += 0.01
    if row['volumen_saliente_30_dias'] > row['volumen_entrante_30_dias'] * 3: prob += 0.01
    if row['numero_transacciones_ultima_hora'] > 5: prob += 0.02
    if row['tiempo_desde_ultima_transaccion'] < 60 and row['importe_transaccion'] > 1000: prob += 0.01
    if row['veces_superar_limite_7_dias'] > 3: prob += 0.02
    if row['metodo_autenticacion'] == 'firma': prob += 0.005
    if row['numero_pin_disponibles'] == 0: prob += 0.02
    if row['tipo_transaccion'] == 'transferencia': prob += 0.01
    if row['tipo_transaccion'] == 'bizum': prob += 0.005
    if row['destino_alto_riesgo'] == 1: prob += 0.04
    return min(prob, 0.95)


def _inyectar_senal(row, rng=None):
    if rng is None:
        rng = np.random.RandomState()
    if rng.random() < 0.55:
        otros = [p for p in PAISES_REGIONES if p != row['customer_country']]
        row['operacion_pais'] = str(rng.choice(otros))
        row['operacion_region'] = str(rng.choice(PAISES_REGIONES[row['operacion_pais']]))
    if rng.random() < 0.60:
        row['dispositivo_reconocido'] = 0
    if rng.random() < 0.35:
        row['estado_tarjeta'] = str(rng.choice(['robada', 'extraviada']))
    if rng.random() < 0.45:
        row['numero_transacciones_ultima_hora'] = int(rng.randint(6, 20))
    if rng.random() < 0.40:
        row['importe_transaccion'] = round(row['limite_importe_transacciones'] * rng.uniform(0.85, 0.99), 2)
    if rng.random() < 0.35:
        row['numero_pin_disponibles'] = 0
    if rng.random() < 0.35:
        row['metodo_autenticacion'] = str(rng.choice(['firma', '3DS']))
    if rng.random() < 0.30:
        row['destino_alto_riesgo'] = 1
    if rng.random() < 0.25:
        row['volumen_saliente_30_dias'] = round(row['volumen_entrante_30_dias'] * rng.uniform(4, 8), 2)
    if rng.random() < 0.30:
        row['tiempo_desde_ultima_transaccion'] = int(rng.randint(5, 55))
    return row


def _aplicar_drift_feature(row, cfg, ronda_idx, rng=None):
    if rng is None:
        rng = np.random.RandomState()
    t = cfg['tipo']
    feature = cfg.get('feature', 'dispositivo_reconocido')

    if t == 'suave':
        inicio = cfg.get('inicio', 20)
        fin = cfg.get('fin', 40)
        magnitud = cfg.get('magnitud', 0.3)
        if inicio <= ronda_idx <= fin:
            progreso = (ronda_idx - inicio) / max(fin - inicio, 1)
            fuerza = progreso * magnitud
            if feature == 'dispositivo_reconocido' and rng.random() < fuerza:
                row['dispositivo_reconocido'] = 0
            elif feature == 'cross_border' and row.get('customer_country'):
                if rng.random() < fuerza:
                    otros = [p for p in PAISES_REGIONES if p != row['customer_country']]
                    row['operacion_pais'] = str(rng.choice(otros))
            elif feature == 'velocidad' and rng.random() < fuerza:
                row['numero_transacciones_ultima_hora'] = int(rng.randint(3, 10))

    elif t == 'abrupto':
        ronda_cambio = cfg.get('ronda', 30)
        if ronda_idx >= ronda_cambio:
            swap_prob = cfg.get('swap_prob', 0.5)
            if feature == 'operacion_pais' and rng.random() < swap_prob:
                otros = [p for p in PAISES_REGIONES if p != row.get('customer_country', 'ES')]
                row['operacion_pais'] = str(rng.choice(otros))
            elif feature == 'saldo' and rng.random() < swap_prob:
                row['saldo_actual'] = round(rng.uniform(10, 100), 2)

    return row



# PIPELINE DE INGRESCIA

class InferencePipeline:
    def __init__(self, model_path):
        obj = joblib.load(model_path)
        self.fe = obj['fe']
        self.scaler = obj['scaler']
        self.imputer = obj['imputer']
        self.lgb_model = obj['lgb_model']
        self.xgb_model = obj['xgb_model']
        self.best_w = obj['best_w']
        self.best_t = obj['best_t']
        self.num_feats = obj['num_feats']
        self.per_channel_thr = obj.get('per_channel_thresholds', {})
        self.metadata = obj.get('metadata', {})

    def predict(self, df):
        X = self.fe.transform(df)
        X = X.drop(columns=['IS_FRAUD'], errors='ignore')
        X_s = X.copy()
        X_s[self.num_feats] = self.scaler.transform(X[self.num_feats])
        X_s[self.num_feats] = self.imputer.transform(X_s[self.num_feats])

        try:
            xgb_cols = self.xgb_model.get_booster().feature_names
            X_s = X_s[xgb_cols]
        except Exception:
            pass

        p_lgb = self.lgb_model.predict_proba(X_s)[:, 1]
        p_xgb = self.xgb_model.predict_proba(X_s)[:, 1]
        y_prob = self.best_w * p_lgb + (1 - self.best_w) * p_xgb

        y_pred = np.zeros(len(y_prob), dtype=int)
        for canal in df['tipo_transaccion'].unique():
            thr = self.per_channel_thr.get(canal, self.best_t)
            mask = df['tipo_transaccion'].values == canal
            y_pred[mask] = (y_prob[mask] >= thr).astype(int)

        return y_prob, y_pred



# MÉTRICAS POR RONDA

def calcular_metricas(y_true, y_prob, y_pred):
    n = len(y_true)
    if n < 2:
        return {'n': n, 'fraudes': int(y_true.sum()),
                'precision': np.nan, 'recall': np.nan, 'f2': np.nan,
                'pr_auc': np.nan, 'roc_auc': np.nan}

    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f2 = fbeta_score(y_true, y_pred, beta=2, zero_division=0)

    prauc = average_precision_score(y_true, y_prob) if y_true.sum() > 0 else np.nan
    roc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        'n': n,
        'fraudes': int(y_true.sum()),
        'tasa_fraude': float(y_true.mean()),
        'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn),
        'precision': round(prec, 4),
        'recall': round(rec, 4),
        'f2': round(f2, 4),
        'pr_auc': round(prauc, 4) if not np.isnan(prauc) else None,
        'roc_auc': round(roc, 4) if not np.isnan(roc) else None,
    }


def calcular_psi(esperado, actual, bins=10):
    """Population Stability Index entre dos distribuciones de score."""
    esperado = np.asarray(esperado).clip(0, 1)
    actual = np.asarray(actual).clip(0, 1)
    edges = np.linspace(0, 1, bins + 1)
    e_counts, _ = np.histogram(esperado, bins=edges)
    a_counts, _ = np.histogram(actual, bins=edges)
    e_pct = e_counts / max(e_counts.sum(), 1)
    a_pct = a_counts / max(a_counts.sum(), 1)
    e_pct = np.clip(e_pct, 1e-6, None)
    a_pct = np.clip(a_pct, 1e-6, None)
    psi = np.sum((a_pct - e_pct) * np.log(a_pct / e_pct))
    return round(float(psi), 4)



# MAIN

def main():
    parser = argparse.ArgumentParser(description='Evaluación por rondas de 100 txns')
    parser.add_argument('--modelo', choices=['v1', 'v2', 'v3'], default='v3',
                        help='Modelo a evaluar (v1, v2, v3)')
    parser.add_argument('--rondas', type=int, default=50,
                        help='Número de rondas de 100 transacciones')
    parser.add_argument('--drift', choices=['baseline', 'suave', 'abrupto', 'concepto'],
                        default='baseline', help='Escenario de drift')
    parser.add_argument('--output', type=str, default='',
                        help='CSV de salida con métricas por ronda')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # Cargar modelo
    model_map = {'v1': 'modelo_07_v1', 'v2': 'modelo_08_v2', 'v3': 'modelo_09_v3'}
    model_name = model_map[args.modelo]
    model_path = Path(__file__).resolve().parent.parent / 'model' / f'{model_name}.pkl'
    if not model_path.exists():
        print(f'ERROR: No se encuentra {model_path}')
        print('Ejecuta primero: python scripts/regenerate_models.py')
        sys.exit(1)

    print(f'Cargando modelo: {model_path.name}')
    pipeline = InferencePipeline(str(model_path))
    print(f'  Dataset:   {pipeline.metadata.get("label", "?")}')
    print(f'  Ensemble:  w(LGB)={pipeline.best_w:.3f} + w(XGB)={1-pipeline.best_w:.3f}')
    print(f'  Threshold: {pipeline.best_t:.4f}')
    if pipeline.per_channel_thr:
        print(f'  Thr/canal: {pipeline.per_channel_thr}')
    print(f'  Features:  {len(pipeline.num_feats)} numéricas')
    print()

    # Configuración de drift
    drift_cfg = None
    if args.drift == 'suave':
        drift_cfg = {'tipo': 'suave', 'inicio': 20, 'fin': 40,
                     'feature': 'dispositivo_reconocido', 'magnitud': 0.4}
    elif args.drift == 'abrupto':
        drift_cfg = {'tipo': 'abrupto', 'ronda': 30,
                     'feature': 'operacion_pais', 'swap_prob': 0.6}
    elif args.drift == 'concepto':
        drift_cfg = {'tipo': 'concepto', 'ronda': 25,
                     'nuevo_patron': 'importe_bajo_madrugada'}

    print(f'Escenario: {args.drift.upper()}')
    print(f'Rondas:    {args.rondas} ({(args.rondas * 100):,} transacciones)')
    print()

    # Baseline de scores (primeras 5 rondas para PSI)
    scores_baseline = []

    # Acumuladores
    y_true_all, y_prob_all, y_pred_all = [], [], []
    resultados_rondas = []

    # Cabecera de tabla
    encabezado = (f'{"Rnda":>4s} {"Fraud":>5s} {"Tasa":>6s} '
                  f'{"Prec":>6s} {"Recall":>6s} {"F2":>6s} '
                  f'{"PR-AUC":>7s} {"ROC-AUC":>7s} {"PSI":>6s} '
                  f'{"TP":>3s} {"FP":>3s} {"FN":>3s}')
    sep = '-' * len(encabezado)
    print(encabezado)
    print(sep)

    for ronda in range(1, args.rondas + 1):
        df_ronda = generar_ronda(n=100, seed=args.seed + ronda,
                                 drift_cfg=drift_cfg, ronda_idx=ronda)
        y_true = df_ronda['IS_FRAUD'].values

        y_prob, y_pred = pipeline.predict(df_ronda)

        y_true_all.extend(y_true.tolist())
        y_prob_all.extend(y_prob.tolist())
        y_pred_all.extend(y_pred.tolist())

        m = calcular_metricas(y_true, y_prob, y_pred)
        m['ronda'] = ronda

        # PSI contra baseline (primeras 5 rondas)
        if ronda <= 5:
            scores_baseline.extend(y_prob.tolist())
            psi = 0.0
        else:
            psi = calcular_psi(scores_baseline, y_prob)

        m['psi'] = psi
        resultados_rondas.append(m)

        # Acumuladas
        m_acum = calcular_metricas(
            np.array(y_true_all), np.array(y_prob_all), np.array(y_pred_all))
        m_acum['ronda'] = ronda

        linea = (f'{ronda:>4d} {m["fraudes"]:>5d} {m["tasa_fraude"]:>6.2%} '
                 f'{m["precision"] if m["precision"] else 0:>6.1%} '
                 f'{m["recall"] if m["recall"] else 0:>6.1%} '
                 f'{m["f2"] if m["f2"] else 0:>6.4f} '
                 f'{m["pr_auc"] if m["pr_auc"] else "-":>7} '
                 f'{m["roc_auc"] if m["roc_auc"] else "-":>7} '
                 f'{psi:>6.4f} '
                 f'{m["tp"]:>3d} {m["fp"]:>3d} {m["fn"]:>3d}')
        print(linea)

        # Detectar drift o degradación
        if ronda > 5:
            advertencias = []
            if m['recall'] is not None and m['recall'] < 0.5:
                advertencias.append(f'recall={m["recall"]:.1%}')
            if psi > 0.1:
                advertencias.append(f'PSI={psi:.4f}')
            if m_acum['recall'] is not None and m_acum['recall'] < 0.8:
                advertencias.append(f'recall acum={m_acum["recall"]:.1%}')
            if advertencias:
                print(f'  [!] Ronda {ronda}: {", ".join(advertencias)}')

    # Resumen final
    print(sep)
    m_final = calcular_metricas(
        np.array(y_true_all), np.array(y_prob_all), np.array(y_pred_all))
    print(f'{"ACUM":>4s} {m_final["fraudes"]:>5d} {m_final["tasa_fraude"]:>6.2%} '
          f'{m_final["precision"] if m_final["precision"] else 0:>6.1%} '
          f'{m_final["recall"] if m_final["recall"] else 0:>6.1%} '
          f'{m_final["f2"] if m_final["f2"] else 0:>6.4f} '
          f'{m_final["pr_auc"] if m_final["pr_auc"] else "-":>7} '
          f'{m_final["roc_auc"] if m_final["roc_auc"] else "-":>7} '
          f'{"":>6s} '
          f'{m_final["tp"]:>3d} {m_final["fp"]:>3d} {m_final["fn"]:>3d}')

    print()
    print(f'Total: {m_final["n"]:,} transacciones | '
          f'{m_final["fraudes"]} fraudes ({m_final["tasa_fraude"]:.2%}) | '
          f'Precision={m_final["precision"]:.1%} | '
          f'Recall={m_final["recall"]:.1%} | '
          f'F2={m_final["f2"]:.4f} | '
          f'PR-AUC={m_final["pr_auc"]:.4f}')

    # Guardar resultados
    if args.output:
        path = Path(args.output)
        df_out = pd.DataFrame(resultados_rondas)
        df_out.to_csv(path, index=False)
        print(f'Resultados guardados: {path.resolve()}')

    # Detección de drift (resumen)
    if args.drift != 'baseline':
        print()
        print('--- Detección de drift ---')
        df_psi = pd.DataFrame(resultados_rondas)
        for fase, inicio, fin in [('Pre-drift', 1, 10),
                                   ('Post-drift', 21, args.rondas)]:
            subset = df_psi[(df_psi['ronda'] >= inicio) & (df_psi['ronda'] <= fin)]
            if len(subset) > 0:
                prec_medio = subset['precision'].mean()
                rec_medio = subset['recall'].mean()
                psi_medio = subset['psi'].mean()
                print(f'  {fase:12s} (rondas {inicio:3d}-{fin:<3d}): '
                      f'Prec={prec_medio:.1%} Rec={rec_medio:.1%} PSI={psi_medio:.4f}')


if __name__ == '__main__':
    main()
