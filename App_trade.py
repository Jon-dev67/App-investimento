from flask import Flask, jsonify
import requests
import os
import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Configuração do logging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

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

# Função para obter preços históricos
def get_historical_prices():
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days=60&interval=daily"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        prices = [entry[1] for entry in data["prices"]]
        return np.array(prices).reshape(-1, 1)
    except Exception as e:
        logging.error(f"❌ Erro ao obter preços históricos: {e}")
        return None

# Criar modelo LSTM
def create_lstm_model():
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(10, 1)),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Treinar o modelo uma vez na inicialização
def train_lstm_once():
    global model, last_sequence

    prices = get_historical_prices()
    if prices is None or len(prices) < 30:
        logging.error("❌ Dados insuficientes para treinar o modelo.")
        return None

    X, y = [], []
    for i in range(len(prices) - 10):
        X.append(prices[i:i+10])
        y.append(prices[i+10])

    X, y = np.array(X), np.array(y)

    model.fit(X, y, epochs=50, batch_size=8, verbose=0)
    last_sequence = X[-1].reshape(1, 10, 1)

# Fazer previsão com o modelo já treinado
def predict_future(days):
    global model, last_sequence
    predictions = []
    sequence = last_sequence.copy()

    for _ in range(days):
        pred = model.predict(sequence)
        predictions.append(float(pred[0, 0]))
        sequence = np.roll(sequence, -1)
        sequence[0, -1, 0] = pred

    return predictions[-1]

# Criar modelo e treinar na inicialização
model = create_lstm_model()
train_lstm_once()

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