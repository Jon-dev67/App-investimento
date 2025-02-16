from flask import Flask, jsonify
import requests
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sqlalchemy import create_engine, text

app = Flask(__name__)

# 🔹 Conectar ao banco SQLite
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///bitcoin.db")
engine = create_engine(DATABASE_URL)

# 🔹 Criar a tabela caso não exista
def init_db():
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS bitcoin_prices (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                price REAL NOT NULL
            )
        """))
        conn.commit()

init_db()  # Criar a tabela ao iniciar a API

# 🔹 Função para buscar preço do Bitcoin
def get_bitcoin_price():
    url = "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT"

    try:
        response = requests.get(url, timeout=5)  # Define tempo limite para evitar travamentos
        response.raise_for_status()  # Lança erro se a resposta não for 200 OK

        data = response.json()
        if "price" in data:
            return float(data["price"])
        else:
            raise KeyError("A resposta da Binance não contém a chave 'price'")

    except requests.exceptions.RequestException as e:
        print(f"❌ Erro ao acessar Binance: {e}")
        return None

# 🔹 Função para salvar o preço no banco
def save_price(price):
    with engine.connect() as conn:
        conn.execute(text("INSERT INTO bitcoin_prices (timestamp, price) VALUES (CURRENT_TIMESTAMP, :price)"), {"price": price})
        conn.commit()

# 🔹 Treinar modelo LSTM para previsões
def train_lstm_model(days):
    df = pd.read_sql("SELECT * FROM bitcoin_prices ORDER BY timestamp ASC", con=engine)

    if len(df) < 30:  # Garantir que há dados suficientes para treinar o modelo
        return None

    # 🔸 Normalizar timestamps e transformar em sequência para LSTM
    df["timestamp"] = pd.to_datetime(df["timestamp"]).astype(int) // 10**9
    prices = df["price"].values.reshape(-1, 1)

    seq_length = 10  # Tamanho da sequência usada para prever o próximo preço
    X, y = [], []
    for i in range(len(prices) - seq_length):
        X.append(prices[i:i+seq_length])
        y.append(prices[i+seq_length])

    X, y = np.array(X), np.array(y)

    # 🔹 Criar modelo LSTM otimizado
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')

    # 🔸 Treinar o modelo com menos épocas para eficiência
    model.fit(X, y, epochs=20, batch_size=8, verbose=0)

    # 🔹 Fazer previsão para os próximos dias
    future_predictions = []
    last_sequence = X[-1].reshape(1, seq_length, 1)

    for _ in range(days):
        pred = model.predict(last_sequence)
        future_predictions.append(float(pred[0, 0]))
        last_sequence = np.roll(last_sequence, -1)
        last_sequence[0, -1, 0] = pred

    return future_predictions[-1]  # Retorna previsão para o último dia da sequência

# 🔹 Endpoint para preço atual do Bitcoin
@app.route("/price", methods=["GET"])
def price():
    price = get_bitcoin_price()
    if price:
        save_price(price)
        return jsonify({"price": price})
    return jsonify({"error": "Não foi possível obter o preço do Bitcoin"}), 500

# 🔹 Endpoint para previsão de 1 dia e 30 dias
@app.route("/predict", methods=["GET"])
def predict():
    prediction_1d = train_lstm_model(1)
    prediction_30d = train_lstm_model(30)

    if prediction_1d and prediction_30d:
        return jsonify({"prediction_1d": prediction_1d, "prediction_30d": prediction_30d})

    return jsonify({"error": "Dados insuficientes para prever"}), 400

# 🔹 Endpoint para recomendações de compra/venda
@app.route("/recommendation", methods=["GET"])
def recommendation():
    price = get_bitcoin_price()
    prediction_1d = train_lstm_model(1)
    prediction_30d = train_lstm_model(30)

    if not prediction_1d or not prediction_30d:
        return jsonify({"error": "Sem dados suficientes para análise"}), 400

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

# 🔹 Rota principal para testar a API
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "API de previsão de Bitcoin está rodando 🚀"}), 200

# 🔹 Iniciar a aplicação corretamente no Render
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Render exige variável de ambiente
    app.run(host="0.0.0.0", port=port, debug=True)