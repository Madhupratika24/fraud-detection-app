from flask import Flask, request, jsonify, render_template_string
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

model  = joblib.load('fraud_detection_model.pkl')
scaler = joblib.load('scaler.pkl')

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Credit Card Fraud Detection</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: Arial;
            background: #0B1F3A;
            color: white;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
        }
        .card {
            background: #12294A;
            padding: 40px;
            border-radius: 16px;
            width: 420px;
            text-align: center;
            box-shadow: 0 8px 32px rgba(0,0,0,0.4);
        }
        h1 { color: #17BEBB; margin-bottom: 8px; font-size: 22px; }
        p  { color: #8aa0b0; margin-bottom: 24px; font-size: 14px; }
        input {
            width: 100%;
            padding: 12px;
            margin: 8px 0;
            border-radius: 8px;
            border: 1px solid #1e3a5f;
            background: #0B1F3A;
            color: white;
            font-size: 15px;
        }
        button {
            width: 100%;
            padding: 14px;
            background: #17BEBB;
            border: none;
            border-radius: 8px;
            color: white;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            margin-top: 16px;
        }
        button:hover { background: #0E7C7B; }
        #result {
            margin-top: 24px;
            padding: 16px;
            border-radius: 8px;
            font-size: 18px;
            font-weight: bold;
            display: none;
        }
        .fraud   { background: #2d0a0a; color: #ff4444; }
        .genuine { background: #0a2d1a; color: #17BEBB; }
        .prob    { font-size: 13px; margin-top: 8px;
                   color: #8aa0b0; font-weight: normal; }
    </style>
</head>
<body>
<div class="card">
    <h1>🔐 Fraud Detection System</h1>
    <p>Credit Card Transaction Checker</p>
    <input type="number" id="amount"
           placeholder="Transaction Amount (€)" />
    <input type="number" id="time"
           placeholder="Time (seconds since first txn)" />
    <button onclick="predict()">
        Analyze Transaction
    </button>
    <div id="result"></div>
</div>

<script>
async function predict() {
    const amount = document.getElementById('amount').value;
    const time   = document.getElementById('time').value;

    if (!amount || !time) {
        alert('Please enter both Amount and Time');
        return;
    }

    const res  = await fetch('/predict', {
        method : 'POST',
        headers: {'Content-Type': 'application/json'},
        body   : JSON.stringify({ amount, time })
    });
    const data = await res.json();
    const div  = document.getElementById('result');
    div.style.display = 'block';

    if (data.prediction === 1) {
        div.className = 'result fraud';
        div.innerHTML =
            `⚠️ FRAUDULENT TRANSACTION
             <div class="prob">
             Fraud: ${data.fraud_prob}% |
             Genuine: ${data.genuine_prob}%
             </div>`;
    } else {
        div.className = 'result genuine';
        div.innerHTML =
            `✅ GENUINE TRANSACTION
             <div class="prob">
             Genuine: ${data.genuine_prob}% |
             Fraud: ${data.fraud_prob}%
             </div>`;
    }
}
</script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML)

@app.route('/predict', methods=['POST'])
def predict():
    data   = request.get_json()
    amount = float(data['amount'])
    time   = float(data['time'])

    # Build input (V1-V28 = 0 for demo)
    features = {f'V{i}': 0.0 for i in range(1, 29)}
    features['Time']   = time
    features['Amount'] = amount

    input_df = pd.DataFrame([features])
    input_df = input_df[['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']] 

    # FIX: use transform() not fit_transform()
    input_df['Amount'] = scaler.transform(input_df[['Amount']].values)
    input_df['Time'] = scaler.transform(input_df[['Time']].values)
    pred  = model.predict(input_df)[0]
    probs = model.predict_proba(input_df)[0]

    return jsonify({
        'prediction'  : int(pred),
        'fraud_prob'  : round(float(probs[1]) * 100, 2),
        'genuine_prob': round(float(probs[0]) * 100, 2)
    })

if __name__ == '__main__':
    app.run(debug=True)