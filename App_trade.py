from flask import Flask, jsonify
import requests
import os
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sqlalchemy import create_engine, text

app = Flask(__name__)

# Conectar ao banco SQLite local
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///bitcoin.db")
engine = create_engine(DATABASE_URL)

# Função para obter preço do Bitcoin
def get_bitcoin_price():
    url = "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT"
    response = requests.get(url)
    data = response.json()
    return float(data["price"])

# Salvar preço no banco
def save_price(price):
    with engine.connect() as conn:
        conn.execute(text("INSERT INTO bitcoin_prices (timestamp, price) VALUES (CURRENT_TIMESTAMP, :price)"), {"price": price})
        conn.commit()

# Treinar modelo LSTM para previsões
def train_lstm_model(days):
    df = pd.read_sql("SELECT * FROM bitcoin_prices ORDER BY timestamp ASC", con=engine)

    if len(df) < 30:  # Garantir que há dados suficientes
        return None

    df["timestamp"] = pd.to_datetime(df["timestamp"]).astype(int) // 10**9
    prices = df["price"].values.reshape(-1, 1)

    # Criar sequências para LSTM
    seq_length = 10
    X, y = [], []
    for i in range(len(prices) - seq_length):
        X.append(prices[i:i+seq_length])
        y.append(prices[i+seq_length])

    X, y = np.array(X), np.array(y)

    # Criar modelo LSTM
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Treinar modelo
    model.fit(X, y, epochs=50, batch_size=8, verbose=0)

    # Fazer previsão para os próximos dias
    future_predictions = []
    last_sequence = X[-1].reshape(1, seq_length, 1)
    for _ in range(days):
        pred = model.predict(last_sequence)
        future_predictions.append(float(pred[0, 0]))
        last_sequence = np.roll(last_sequence, -1)
        last_sequence[0, -1, 0] = pred

    return future_predictions[-1]  # Retorna a previsão para o último dia da sequência

# Endpoint para preço atual
@app.route("/price", methods=["GET"])
def price():
    price = get_bitcoin_price()
    save_price(price)
    return jsonify({"price": price})

# Endpoint para previsão de 1 dia e 30 dias
@app.route("/predict", methods=["GET"])
def predict():
    prediction_1d = train_lstm_model(1)
    prediction_30d = train_lstm_model(30)
    
    if prediction_1d and prediction_30d:
        return jsonify({"prediction_1d": prediction_1d, "prediction_30d": prediction_30d})
    
    return jsonify({"error": "Dados insuficientes para prever."})

# Endpoint para recomendações de compra/venda
@app.route("/recommendation", methods=["GET"])
def recommendation():
    price = get_bitcoin_price()
    prediction_1d = train_lstm_model(1)
    prediction_30d = train_lstm_model(30)

    if not prediction_1d or not prediction_30d:
        return jsonify({"error": "Sem dados suficientes para análise."})

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

if __name__ == "__main__":
    app.run(debug=True)