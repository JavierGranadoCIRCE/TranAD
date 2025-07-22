
import os
import pandas as pd
import numpy as np
from itertools import combinations_with_replacement
from sklearn.model_selection import train_test_split

# Configuración
input_dir = "./original_csvs"
output_train_dir = "./siamese_dataset/train"
output_test_dir = "./siamese_dataset/test"
os.makedirs(output_train_dir, exist_ok=True)
os.makedirs(output_test_dir, exist_ok=True)

# Cargar todos los archivos CSV
csv_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".csv")])
csv_data = {}

for f in csv_files:
    df = pd.read_csv(os.path.join(input_dir, f))
    df = df.drop(columns=["t (s)"])  # Eliminar columna de tiempo
    csv_data[f] = df

# Generar todas las combinaciones con repetición
combinations = list(combinations_with_replacement(csv_files, 2))

samples = []
for idx, (f1, f2) in enumerate(combinations, start=1):
    df1 = csv_data[f1].reset_index(drop=True)
    df2 = csv_data[f2].reset_index(drop=True)

    # Usar el mínimo número de filas entre los dos para emparejar
    min_len = min(len(df1), len(df2))
    df1 = df1.iloc[:min_len]
    df2 = df2.iloc[:min_len]

    combined = pd.concat([df1.add_prefix("signal 1_"), df2.add_prefix("signal 2_")], axis=1)
    combined["label"] = 0 if f1 == f2 else 1
    samples.append((idx, combined))

# Separar en entrenamiento y test
ids = [i for i, _ in samples]
train_ids, test_ids = train_test_split(ids, test_size=0.2, random_state=42)

# Guardar los archivos
for idx, df in samples:
    if idx in train_ids:
        df.to_csv(os.path.join(output_train_dir, f"{idx}.csv"), index=False)
    else:
        df["threshold"] = 0.5
        df.to_csv(os.path.join(output_test_dir, f"{idx}.csv"), index=False)
