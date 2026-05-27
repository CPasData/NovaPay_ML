"""
Genera un CSV con los datos originales más predicciones en formato compacto.

Lee un CSV de transacciones, ejecuta el pipeline completo, y guarda un CSV
con las columnas originales + los 10 campos de predicción.

Uso:
    python scripts/prediccion_lote.py --input data/lote_sin_target.csv --output data/predicciones.csv
    python scripts/prediccion_lote.py --input data/muestra_sin_etiqueta.csv --output data/predicciones.csv --modelo v2
"""

import sys, joblib, warnings, argparse
warnings.filterwarnings('ignore')
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_engineering import FeatureEngineer

FE_CAMPOS = {
    'es_transfronteriza': 'cross_border',
    'ratio_imp_limite':   'txn_vs_limit_pct',
    'intensidad_tx':      'txn_intensity',
    'severidad_tx':       'txn_severity',
    'flujo_neto_30d':     'net_flow_30d',
}


def calcular_impacto(is_fraud, importe):
    if is_fraud == 0:
        return 0
    if importe < 500:
        return 1
    if importe < 2000:
        return 2
    return 3


def generar_mensaje(prob):
    pct = round(prob * 100)
    if prob >= 0.5:
        return f"Fraude detectado - probabilidad fraude {pct}%"
    return f"Transaccion legitima - probabilidad fraude {pct}%"


def main():
    parser = argparse.ArgumentParser(description='Batch inference: datos originales + predicciones compactas')
    parser.add_argument('--input', required=True, help='CSV de entrada con transacciones')
    parser.add_argument('--output', required=True, help='CSV de salida con predicciones')
    parser.add_argument('--modelo', choices=['v1', 'v2', 'v3'], default='v3',
                        help='Modelo a usar (v1=original, v2=mejorado, v3=3por100)')
    args = parser.parse_args()

    model_map = {'v1': 'modelo_07_v1', 'v2': 'modelo_08_v2', 'v3': 'modelo_09_v3'}
    model_name = model_map[args.modelo]
    model_path = Path(__file__).resolve().parent.parent / 'model' / f'{model_name}.pkl'
    if not model_path.exists():
        print(f'ERROR: No se encuentra {model_path}')
        sys.exit(1)

    print(f'Cargando modelo: {model_path.name}')
    obj = joblib.load(model_path)

    fe = obj['fe']
    scaler = obj['scaler']
    imputer = obj['imputer']
    lgb_model = obj['lgb_model']
    xgb_model = obj['xgb_model']
    best_w = obj['best_w']
    best_t = obj['best_t']
    num_feats = obj['num_feats']
    per_channel_thr = obj.get('per_channel_thresholds', {})

    print(f'Ensemble: w(LGB)={best_w:.3f} + w(XGB)={1-best_w:.3f}')
    print(f'Threshold F2: {best_t:.4f}')
    if per_channel_thr:
        print(f'Thresholds por canal: {per_channel_thr}')
    print()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f'ERROR: No se encuentra {input_path}')
        sys.exit(1)

    print(f'Leyendo: {input_path}')
    df = pd.read_csv(input_path)
    n_total = len(df)
    print(f'Registros: {n_total:,}')

    # Feature engineering
    print('Ejecutando pipeline de inferencia...')
    X = fe.transform(df)

    # Extraer campos calculados antes de imputer/scaler
    for nombre, src in FE_CAMPOS.items():
        if src in X.columns:
            vals = X[src].values
            if nombre == 'es_transfronteriza':
                df[nombre] = vals.astype(int)
            else:
                df[nombre] = vals.astype(float).round(4)
        else:
            df[nombre] = 0

    X = X.drop(columns=['IS_FRAUD'], errors='ignore')

    X_s = X.copy()
    X_s[num_feats] = scaler.transform(X[num_feats])
    X_s[num_feats] = imputer.transform(X_s[num_feats])

    # Alinear columnas al orden exacto de entrenamiento
    try:
        xgb_cols = xgb_model.get_booster().feature_names
        X_s = X_s[xgb_cols]
    except Exception:
        pass
    try:
        lgb_cols = lgb_model.booster_.feature_name()
        X_s = X_s[lgb_cols]
    except Exception:
        pass

    p_lgb = lgb_model.predict_proba(X_s)[:, 1]
    p_xgb = xgb_model.predict_proba(X_s)[:, 1]
    y_prob = best_w * p_lgb + (1 - best_w) * p_xgb

    # Threshold por canal o global
    y_pred = np.zeros(len(y_prob), dtype=int)
    for canal in df['tipo_transaccion'].unique():
        thr_c = per_channel_thr.get(canal, best_t)
        mask = df['tipo_transaccion'].values == canal
        y_pred[mask] = (y_prob[mask] >= thr_c).astype(int)

    # Columnas de predicción
    df['is_fraud'] = y_pred
    df['prob_fraud'] = np.round(y_prob, 4)
    df['impacto_fraude'] = [calcular_impacto(pred, imp)
                            for pred, imp in zip(y_pred, df['importe_transaccion'])]
    df['mensaje'] = [generar_mensaje(p) for p in y_prob]

    # Reordenar: originales + predicciones al final
    base_cols = [c for c in df.columns if c not in (
        'is_fraud', 'prob_fraud', 'impacto_fraude', 'mensaje',
        *FE_CAMPOS.keys())]
    pred_cols = ['is_fraud', 'prob_fraud', 'impacto_fraude',
                 *FE_CAMPOS.keys(), 'mensaje']
    df = df[base_cols + pred_cols]

    output_path = Path(args.output)
    output_path.parent.mkdir(exist_ok=True)
    df.to_csv(output_path, index=False, encoding='utf-8')

    detectados = y_pred.sum()
    print()
    print(f'Predicciones: {detectados} fraudes detectados ({detectados/n_total*100:.2f}%)')
    tiene_label = 'IS_FRAUD' in df.columns
    if tiene_label:
        reales = df['IS_FRAUD'].sum()
        tp = ((y_pred == 1) & (df['IS_FRAUD'] == 1)).sum()
        print(f'  TP={tp}  FN={reales - tp}  FP={detectados - tp}  TN={n_total - reales - detectados + tp}')
    print(f'Guardado: {output_path.resolve()}')


if __name__ == '__main__':
    main()
