from flask import Flask, request, jsonify, render_template
import requests  

app = Flask(__name__)

FLASK_API_URL = "http://127.0.0.1:5000/predict"  

@app.route('/')
def home():
    return render_template('interface.html')  

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    features = [
        int(request.form['number_of_adults']),
        int(request.form['number_of_children']),
        int(request.form['number_of_weekend_nights']),
        int(request.form['number_of_week_nights']),
        request.form['type_of_meal'],
        int(request.form['car_parking_space']),
        request.form['room_type'],
        int(request.form['lead_time']),
        request.form['market_segment_type'],
        int(request.form['repeated']),
        int(request.form['p_c']),
        int(request.form['p_not_c']),
        float(request.form['average_price']),
        int(request.form['special_requests']),
        request.form['date_of_reservation']
    ]

    data = {"features": features}
    
    response = requests.post(FLASK_API_URL, json=data)
    
    if response.status_code == 200:
        result = response.json()['prediction']
    else:
        result = "Error in prediction"

    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
