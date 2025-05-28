from flask import Flask, render_template, request
import joblib
import pandas as pd


app = Flask(__name__)

model = None 
r2_score = None
def loadModel():
    global model, r2_score
    if model is None and r2_score is None:
        model = joblib.load('./model/housePriceLinearModel.pkl')
        r2_score = joblib.load('./model/R2_Score.pkl')

@app.before_request
def init():
    loadModel()
    
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    area = float(request.form['area'])
    bedrooms = int(request.form['bedrooms'])
    bathrooms = int(request.form['bathrooms'])
    stories = int(request.form['stories'])
    mainroad = request.form['mainroad']
    guestroom = request.form['guestroom']
    basement = request.form['basement']
    hotwaterheating = request.form['hotwaterheating']
    airconditioning = request.form['airconditioning']
    parking = int(request.form['parking'])
    prefarea = request.form['prefarea']
    furnishingstatus = request.form['furnishingstatus']

    data = pd.DataFrame({
        'area': [area],
        'bedrooms': [bedrooms],
        'bathrooms': [bathrooms],
        'stories': [stories],
        'mainroad': [mainroad],
        'guestroom': [guestroom],
        'basement': [basement],
        'hotwaterheating': [hotwaterheating],
        'airconditioning': [airconditioning],
        'parking': [parking],
        'prefarea': [prefarea],
        'furnishingstatus': [furnishingstatus]
    })

    prediction = model.predict(data)[0]
    prediction = round(prediction)

    return render_template('result.html', res=f'{prediction}', r2_score=f"{r2_score*100:.2f}")

if __name__=='__main__':
    app.run(debug=True)