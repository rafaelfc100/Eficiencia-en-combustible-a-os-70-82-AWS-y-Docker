import joblib
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)

# Configurar CORS específicamente para tu frontend
CORS(app, resources={
    r"/*": {
        "origins": ["http://52.14.154.121", "http://localhost"],
        "methods": ["GET", "POST"],
        "allow_headers": ["Content-Type"]
    }
})

# Cargar scaler y modelo
scaler = joblib.load("model/scaler.pkl")
model = joblib.load("model/model.pkl")

# Mapeo de etiquetas a texto
label_map = {
    0: "Auto más eficiente, bajo consumo de combustible",
    1: "Auto menos eficiente, alto consumo de combustible"
}

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json["features"]
        data = np.array(data).reshape(1, -1)

        # Escalar antes de predecir
        data_scaled = scaler.transform(data)

        pred = model.predict(data_scaled)
        texto_pred = label_map[int(pred[0])]
        return jsonify({"prediction": texto_pred})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/")
def home():
    return "API funcionando correctamente"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)