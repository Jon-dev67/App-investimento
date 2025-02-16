from flask import Flask, jsonify
import requests
import os
import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta

# Configuração do logging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

# Carregar modelo treinado
try:
    model = load_model("lstm_model.h5")
    logging.info("✅ Modelo LSTM carregado com sucesso!")
except Exception as e:
    logging.error(f"❌ Erro ao carregar modelo: {e}")
    model = None

# Cache de previsões para evitar recálculo frequente
cache = {
    "timestamp": None,
    "predictions": None
}

# Função para obter o preço do Bitcoin
def get_bitcoin_price():
    url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        return float(data["bitcoin"]["usd"])
    except Exception as e:
        logging.error(f"❌ Erro ao obter preço do Bitcoin: {e}")
        return None

# Fazer previsão usando o modelo carregado
def predict_future(days):
    global cache

    # Se a previsão foi feita nos últimos 30 minutos, usar cache
    if cache["timestamp"] and (datetime.now() - cache["timestamp"]) < timedelta(minutes=30):
        logging.info("🟢 Usando previsão do cache")
        return cache["predictions"].get(days)

    if model is None:
        return None

    # Pegar preços históricos recentes
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days=60&interval=daily"
    response = requests.get(url)
    data = response.json()
    prices = [entry[1] for entry in data["prices"]]
    prices = np.array(prices[-10:]).reshape(1, 10, 1)

    predictions = {}
    sequence = prices.copy()

    for d in [1, 30]:  # Prever 1 dia e 30 dias
        pred_values = []
        for _ in range(d):
            pred = model.predict(sequence)
            pred_values.append(float(pred[0, 0]))
            sequence = np.roll(sequence, -1)
            sequence[0, -1, 0] = pred
        predictions[d] = pred_values[-1]

    # Atualizar cache
    cache["timestamp"] = datetime.now()
    cache["predictions"] = predictions

    return predictions.get(days)

# Rota para checagem de saúde
@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"message": "API rodando 🚀"}), 200

# Endpoint para previsão
@app.route("/predict", methods=["GET"])
def predict():
    prediction_1d = predict_future(1)
    prediction_30d = predict_future(30)

    if prediction_1d and prediction_30d:
        return jsonify({"prediction_1d": prediction_1d, "prediction_30d": prediction_30d})

    return jsonify({"error": "Erro ao prever preços."}), 500

# Endpoint de recomendação
@app.route("/recommendation", methods=["GET"])
def recommendation():
    price = get_bitcoin_price()
    if price is None:
        return jsonify({"error": "Erro ao obter preço."}), 500

    prediction_1d = predict_future(1)
    prediction_30d = predict_future(30)

    decision = "Aguardar"
    if prediction_1d > price and prediction_30d > price:
        decision = "Comprar"
    elif prediction_1d < price and prediction_30d < price:
        decision = "Vender"

    return jsonify({
        "recommendation": decision,
        "current_price": price,
        "predicted_price_1d": prediction_1d,
        "predicted_price_30d": prediction_30d
    })

# Iniciar API no Render
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=True)