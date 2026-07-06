from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load model and encoders
model = pickle.load(open('model.pkl', 'rb'))
charging_mode_enc = pickle.load(open('Charging Mode.pkl', 'rb'))
battery_type_enc = pickle.load(open('Battery Type.pkl', 'rb'))
ev_model_enc = pickle.load(open('EV Model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    soc = float(request.form['soc'])
    voltage = float(request.form['voltage'])
    current = float(request.form['current'])
    batt_temp = float(request.form['batt_temp'])
    amb_temp = float(request.form['amb_temp'])
    duration = float(request.form['duration'])
    degradation = float(request.form['degradation'])
    cycles = int(request.form['cycles'])
    optimal_class = int(request.form['optimal_class'])

    charging_mode = charging_mode_enc.transform([request.form['charging_mode']])[0]
    battery_type = battery_type_enc.transform([request.form['battery_type']])[0]
    ev_model = ev_model_enc.transform([request.form['ev_model']])[0]

    features = np.array([[soc, voltage, current, batt_temp, amb_temp,
                          duration, degradation, charging_mode,
                          battery_type, cycles, ev_model, optimal_class]])

    prediction = model.predict(features)[0]

    return render_template('result.html', prediction=round(prediction, 2))

if __name__ == '__main__':
    app.run(debug=True)

