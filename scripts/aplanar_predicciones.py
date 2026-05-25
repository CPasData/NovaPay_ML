"""
Lee predicciones.json (generado por separar_prediccion_json.py) y genera un JSON
aplanado con los campos clave del modelo para cada transaccion.

Uso:
    python scripts/aplanar_predicciones.py --input data/predicciones.json --output data/predicciones_aplanadas.json
"""

import argparse, json
from pathlib import Path


def generar_mensaje(prob, is_fraud):
    pct = round(prob * 100)
    if is_fraud:
        return f"Fraude detectado - probabilidad fraude {pct}%"
    return f"Transaccion legitima - probabilidad fraude {pct}%"


CAMPO_ORIGEN = {
    'es_transfronteriza': 'cross_border',
    'ratio_imp_limite': 'txn_vs_limit_pct',
    'intensidad_tx': 'txn_intensity',
    'severidad_tx': 'txn_severity',
    'flujo_neto_30d': 'net_flow_30d',
}


def main():
    parser = argparse.ArgumentParser(description='Aplanar predicciones.json a formato compacto')
    parser.add_argument('--input', required=True, help='predicciones.json generado por separar_prediccion_json.py')
    parser.add_argument('--output', default='data/predicciones_aplanadas.json', help='JSON de salida')
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f'ERROR: No se encuentra {input_path}')
        return

    with open(input_path, 'r', encoding='utf-8') as f:
        datos = json.load(f)

    print(f'Leidos {len(datos)} registros de {input_path}')

    resultado = []
    faltan = set()
    for reg in datos:
        prob = reg.get('probabilidad_fraude', 0)
        pred = reg.get('prediccion_fraude', 0)
        salida = {
            'id_transaccion': reg.get('id_transaccion', ''),
            'is_fraud': int(pred) if pred is not None else 0,
            'prob_fraud': round(float(prob), 4) if prob is not None else 0.0,
            'impacto_fraude': int(reg.get('impacto_fraude', 0) or 0),
            'es_transfronteriza': int(reg.get('cross_border', 0) or 0),
            'ratio_imp_limite': round(float(reg.get('txn_vs_limit_pct', 0) or 0), 4),
            'intensidad_tx': round(float(reg.get('txn_intensity', 0) or 0), 4),
            'severidad_tx': round(float(reg.get('txn_severity', 0) or 0), 4),
            'flujo_neto_30d': round(float(reg.get('net_flow_30d', 0) or 0), 2),
            'mensaje': generar_mensaje(prob, pred),
        }
        resultado.append(salida)

        for dest, src in CAMPO_ORIGEN.items():
            if src not in reg:
                faltan.add(src)

    output_path = Path(args.output)
    output_path.parent.mkdir(exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(resultado, f, indent=2, ensure_ascii=False)

    print(f'Generados {len(resultado)} registros aplanados')
    print(f'Guardado en: {output_path.resolve()}')

    if faltan:
        print(f'AVISO: columnas no encontradas en origen: {sorted(faltan)}')


if __name__ == '__main__':
    main()
