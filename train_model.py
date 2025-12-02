import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l2

import matplotlib.pyplot as plt
import seaborn as sns

# =====================================
# 0. Semillas
# =====================================
np.random.seed(42)
tf.random.set_seed(42)

# =====================================
# 1. Carpetas
# =====================================
os.makedirs("model", exist_ok=True)
os.makedirs("plots", exist_ok=True)

# =====================================
# 2. Cargar dataset
# =====================================
df = pd.read_csv("model/data/auto-mpg.csv")
df["horsepower"] = pd.to_numeric(df["horsepower"], errors="coerce")
df = df.dropna(subset=["horsepower"])

mpg_promedio = df["mpg"].mean()
df["label"] = df["mpg"].apply(lambda x: 1 if x >= mpg_promedio else 0)

df = df.drop(columns=["car_name", "mpg"])

# =====================================
# 3. Separación
# =====================================
X = df.drop(columns=["label"])
y = df["label"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =====================================
# 4. Escalado
# =====================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =====================================
# 5. Modelo anti-overfitting
# =====================================
model = Sequential([
    Dense(32, activation='relu', kernel_regularizer=l2(0.001),
          input_shape=(X_train_scaled.shape[1],)),
    BatchNormalization(),
    Dropout(0.2),

    Dense(16, activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.2),

    Dense(8, activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.1),

    Dense(1, activation='sigmoid')
])

optimizer = Adam(learning_rate=0.0003)

model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print(model.summary())

# =====================================
# 6. Callbacks Anti-overfit
# =====================================
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    filepath="model/mejor_modelo.keras",
    monitor="val_loss",
    save_best_only=True
)

# =====================================
# 7. Entrenamiento
# =====================================
history = model.fit(
    X_train_scaled, y_train,
    validation_data=(X_test_scaled, y_test),
    epochs=120,
    batch_size=16,
    callbacks=[early_stop, checkpoint],
    verbose=1
)

# =====================================
# 8. Evaluación
# =====================================
y_pred = (model.predict(X_test_scaled) > 0.5).astype(int)

acc = accuracy_score(y_test, y_pred)
print(f"\nAccuracy en Test: {acc:.4f}\n")

print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
plt.title("Matriz de Confusión")
plt.tight_layout()
plt.savefig("plots/matriz_confusion.png")
plt.close()

# =====================================
# 9. Gráficas
# =====================================
plt.figure(figsize=(9, 5))
plt.plot(history.history['loss'], label="Train Loss")
plt.plot(history.history['val_loss'], label="Test Loss")
plt.title("Curva de Pérdida")
plt.xlabel("Épocas")
plt.ylabel("Loss")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("plots/loss_curve.png")
plt.close()

plt.figure(figsize=(9, 5))
plt.plot(history.history['accuracy'], label="Train Acc")
plt.plot(history.history['val_accuracy'], label="Test Acc")
plt.title("Curva de Accuracy")
plt.xlabel("Épocas")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("plots/accuracy_curve.png")
plt.close()

# =====================================
# 10. Guardado
# =====================================
model.save("model/model_tf.keras")
joblib.dump(scaler, "model/scaler.pkl")

print("\n============== GUARDADO ==============")
print("✔ Mejor modelo: model/mejor_modelo.keras")
print("✔ Modelo final: model/model_tf.keras")
print("✔ Scaler: model/scaler.pkl")
print("✔ Gráficas en /plots/")
