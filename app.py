from flask import Flask, request, jsonify
import numpy as np
import joblib
from perceptron import Perceptron, train_model

app = Flask(__name__)

model, scaler = train_model()
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    data = np.array(data['data'])
    data_std = scaler.transform(data)
    prediction = model.predict(data_std)
    return jsonify({"result": prediction.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)