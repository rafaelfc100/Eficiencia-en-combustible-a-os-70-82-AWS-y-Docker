import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# ========================================================
# 0. Carpetas
# ========================================================
os.makedirs("eda_plots", exist_ok=True)

# ========================================================
# 1. Cargar dataset
# ========================================================
df = pd.read_csv("model/data/auto-mpg.csv")

# horsepower a numérico
df["horsepower"] = pd.to_numeric(df["horsepower"], errors="coerce")

# ========================================================
# 2. Información general
# ========================================================
print("\n===== HEAD =====")
print(df.head())

print("\n===== INFO =====")
print(df.info())

print("\n===== DESCRIPCIÓN =====")
print(df.describe())

print("\n===== NULOS =====")
print(df.isna().sum())

# ========================================================
# 3. Distribuciones
# ========================================================
for col in df.columns:
    if df[col].dtype != "object":
        plt.figure(figsize=(7, 4))
        sns.histplot(df[col], kde=True)
        plt.title(f"Distribución de {col}")
        plt.tight_layout()
        plt.savefig(f"eda_plots/dist_{col}.png")
        plt.close()

# ========================================================
# 4. Boxplots (detección de outliers)
# ========================================================
num_cols = df.select_dtypes(include=np.number).columns

for col in num_cols:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot de {col}")
    plt.tight_layout()
    plt.savefig(f"eda_plots/box_{col}.png")
    plt.close()

# ========================================================
# 5. Matriz de correlación
# ========================================================
plt.figure(figsize=(10, 8))
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Matriz de Correlación")
plt.tight_layout()
plt.savefig("eda_plots/correlacion.png")
plt.close()

# ========================================================
# 6. Relaciones con mpg
# ========================================================
plt.figure(figsize=(10, 8))
for col in num_cols:
    if col != "mpg":
        sns.scatterplot(x=df[col], y=df["mpg"])
        plt.title(f"{col} vs MPG")
        plt.tight_layout()
        plt.savefig(f"eda_plots/scatter_{col}_mpg.png")
        plt.close()

print("\n===== EDA COMPLETO =====")
print("Gráficas guardadas en: eda_plots/")
