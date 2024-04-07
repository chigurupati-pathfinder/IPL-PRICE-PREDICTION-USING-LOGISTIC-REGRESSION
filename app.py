from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Load the data and train the model
data = pd.read_csv(r'C:\Users\SOUBHAGYALAKSHMI CH\Documents\ML machine learning\IPL IMB381IPL2013.csv')
X = data[['AGE', 'T-RUNS', 'T-WKTS', 'ODI-RUNS-S', 'ODI-SR-B', 'ODI-WKTS', 'ODI-SR-BL', 'CAPTAINCY EXP', 'RUNS-S', 'HS', 'AVE', 'SR-B', 'SIXERS', 'RUNS-C', 'WKTS', 'AVE-BL', 'ECON', 'SR-BL', 'AUCTION YEAR']]
y = data['BASE PRICE']
model = LinearRegression()
model.fit(X, y)

# Define a route to render the HTML form
@app.route('/')
def index():
    return render_template('index.html')

# Define a route to handle form submission and predict price
@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from form
    features = ['AGE', 'T-RUNS', 'T-WKTS', 'ODI-RUNS-S', 'ODI-SR-B', 'ODI-WKTS', 'ODI-SR-BL', 'CAPTAINCY EXP', 'RUNS-S', 'HS', 'AVE', 'SR-B', 'SIXERS', 'RUNS-C', 'WKTS', 'AVE-BL', 'ECON', 'SR-BL', 'AUCTION YEAR']
    input_data = [float(request.form[feature]) for feature in features]
    
    # Predict player price
    predicted_price = model.predict([input_data])[0]

    return render_template('result.html', predicted_price=predicted_price)

if __name__ == '__main__':
    app.run(debug=True)