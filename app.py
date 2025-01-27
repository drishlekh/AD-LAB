# from flask import Flask, render_template, request, jsonify
# import pandas as pd
# from datetime import datetime
# from model.lstm_model import StockPredictor

# app = Flask(__name__)

# # Initialize and train the model
# print("Initializing and training the model...")
# predictor = StockPredictor()
# predictor.train_model()
# print("Model training completed!")

# # Load the dataset to get available dates
# df = pd.read_csv('model/AAPL.csv')
# df['date'] = pd.to_datetime(df['date'])
# min_date = df['date'].min().strftime('%Y-%m-%d')
# max_date = df['date'].max().strftime('%Y-%m-%d')

# @app.route('/')
# def home():
#     return render_template('index.html',
#                          min_date=min_date,
#                          max_date=max_date)

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         selected_date = request.form['date']
#         selected_date = pd.to_datetime(selected_date)

#         # Get actual price
#         actual_price_row = df[df['date'].dt.date == selected_date.date()]
#         actual_price = actual_price_row['close'].values[0] if not actual_price_row.empty else None

#         # Get predicted price
#         predicted_price = predictor.predict_price(selected_date)

#         if predicted_price is None or actual_price is None:
#             return jsonify({
#                 'error': 'Unable to make prediction for the selected date'
#             }), 400

#         return jsonify({
#             'predicted_price': round(float(predicted_price), 2),
#             'actual_price': round(float(actual_price), 2)
#         })
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# if __name__ == '__main__':
#     app.run(debug=True)




from flask import Flask, render_template, request, jsonify
import pandas as pd
from datetime import datetime
from model.lstm_model import StockPredictor
from model.linearReg_model import StockPredictorLR

app = Flask(__name__)

# Initialize and train both models
print("Initializing and training models...")
lstm_predictor = StockPredictor()
lr_predictor = StockPredictorLR()

print("Training LSTM model...")
lstm_predictor.train_model()
print("Training Linear Regression model...")
lr_predictor.train_model()
print("Model training completed!")

# Load the dataset to get available dates
df = pd.read_csv('model/AAPL.csv')
df['date'] = pd.to_datetime(df['date'])
min_date = df['date'].min().strftime('%Y-%m-%d')
max_date = df['date'].max().strftime('%Y-%m-%d')

@app.route('/')
def home():
    return render_template('index.html',
                         min_date=min_date,
                         max_date=max_date)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        selected_date = request.form['date']
        model_type = request.form['model_type']
        selected_date = pd.to_datetime(selected_date)

        # Get actual price
        actual_price_row = df[df['date'].dt.date == selected_date.date()]
        actual_price = actual_price_row['close'].values[0] if not actual_price_row.empty else None

        # Get predicted price based on model type
        if model_type == 'lstm':
            predicted_price = lstm_predictor.predict_price(selected_date)
        else:  # linear regression
            predicted_price = lr_predictor.predict_price(selected_date)

        if predicted_price is None or actual_price is None:
            return jsonify({
                'error': 'Unable to make prediction for the selected date'
            }), 400

        return jsonify({
            'predicted_price': round(float(predicted_price), 2),
            'actual_price': round(float(actual_price), 2)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)