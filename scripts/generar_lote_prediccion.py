"""
Genera un CSV combinando dataset_fraude.csv y dataset_fraude_mejorado.csv
SIN columnas target (IS_FRAUD, IMPACTO_FRAUDE) para pasar por prediccion_lote.py.

Uso:
    python scripts/generar_lote_prediccion.py
    python scripts/generar_lote_prediccion.py --output data/lote_sin_target.csv
"""

import argparse
import pandas as pd
from pathlib import Path

COLUMNS_EXCLUIR = ['IS_FRAUD', 'IMPACTO_FRAUDE']

def main():
    parser = argparse.ArgumentParser(description='Generar CSV sin targets para predicción')
    parser.add_argument('--output', type=str, default='data/lote_sin_target.csv',
                        help='Ruta del CSV de salida')
    args = parser.parse_args()

    base = Path(__file__).resolve().parent.parent
    data_dir = base / 'data'

    rutas = [
        ('v1', data_dir / 'dataset_fraude.csv'),
        ('v2', data_dir / 'dataset_fraude_mejorado.csv'),
    ]

    todos = []
    for nombre, ruta in rutas:
        df = pd.read_csv(ruta)
        print(f'{nombre}: {len(df):,} filas, {len(df.columns)} columnas')
        for col in COLUMNS_EXCLUIR:
            if col in df.columns:
                df = df.drop(columns=[col])
        todos.append(df)

    df_out = pd.concat(todos, ignore_index=True)
    print(f'\nCombinado: {len(df_out):,} filas ({len(df_out.columns)} columnas, sin targets)')

    output_path = Path(args.output)
    output_path.parent.mkdir(exist_ok=True)
    df_out.to_csv(output_path, index=False, encoding='utf-8')
    print(f'Guardado: {output_path.resolve()}')

    print(f'\nPara generar predicciones ejecuta:')
    print(f'  python scripts/prediccion_lote.py --input {args.output} --output data/predicciones_combinadas.csv --modelo v2')

if __name__ == '__main__':
    main()
