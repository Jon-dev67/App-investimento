# API de Previsão de Preço de Bitcoin

Esta é uma API simples construída com Flask que fornece previsões do preço do Bitcoin usando um modelo de aprendizado de máquina (LSTM). A API permite acessar o preço atual do Bitcoin, realizar previsões para os próximos 1 e 30 dias, e obter recomendações de compra/venda com base nas previsões.

## Endpoints

### 1. **`GET /price`**
   Retorna o preço atual do Bitcoin.

   **Resposta**:
   ```json
   {
     "price": "valor_atual"
   }
