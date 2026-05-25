"""
Lee un CSV generado por prediccion_lote.py y lo separa en dos JSON:
  - transacciones.json:   columnas originales (datos crudos de entrada)
  - predicciones.json:    columnas de feature engineering + predicciones
                          (incluye id_transaccion para enlazar)

Uso:
    python scripts/separar_prediccion_json.py --input data/predicciones_v2.csv --output-dir data/
"""

import argparse, json
from pathlib import Path
import pandas as pd

PRED_COLS = ['probabilidad_fraude', 'prediccion_fraude', 'impacto_fraude']

ORIGINAL_COLS = [
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
    'IS_FRAUD', 'IMPACTO_FRAUDE',
]


def main():
    parser = argparse.ArgumentParser(description='Separar CSV de predicciones en dos JSON')
    parser.add_argument('--input', required=True, help='CSV generado por prediccion_lote.py')
    parser.add_argument('--output-dir', default='data', help='Directorio de salida para los JSON')
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f'ERROR: No se encuentra {input_path}')
        return

    df = pd.read_csv(input_path)
    print(f'Leidas {len(df)} filas, {len(df.columns)} columnas')

    # Identificar grupos de columnas
    cols_disponibles = set(df.columns)

    raw_cols = [c for c in ORIGINAL_COLS if c in cols_disponibles]
    pred_cols = [c for c in PRED_COLS if c in cols_disponibles]
    fe_cols = [c for c in df.columns if c not in raw_cols and c not in pred_cols]

    # Asegurar que id_transaccion esta en predicciones
    if 'id_transaccion' in cols_disponibles and 'id_transaccion' not in fe_cols:
        fe_cols = ['id_transaccion'] + fe_cols + pred_cols
    else:
        fe_cols = fe_cols + pred_cols

    df_raw = df[raw_cols]
    df_fe = df[fe_cols]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    path_txn = output_dir / 'transacciones.json'
    path_pred = output_dir / 'predicciones.json'

    with open(path_txn, 'w', encoding='utf-8') as f:
        json.dump(df_raw.to_dict(orient='records'), f, indent=2, default=str, ensure_ascii=False)

    with open(path_pred, 'w', encoding='utf-8') as f:
        json.dump(df_fe.to_dict(orient='records'), f, indent=2, default=str, ensure_ascii=False)

    print(f'Transacciones -> {path_txn.resolve()} ({len(df_raw)} registros, {len(raw_cols)} columnas)')
    print(f'Predicciones  -> {path_pred.resolve()} ({len(df_fe)} registros, {len(fe_cols)} columnas)')
    print(f'  - FE features: {len(fe_cols) - len(pred_cols) - 1}')
    print(f'  - Prediccion:  {len(pred_cols)} columnas')


if __name__ == '__main__':
    main()
