# Create API of ML model using flask

# Import libraries
import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load the model
model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def index():
    return render_template('request.html')

@app.route('/api', methods=['POST'])
def predict():
    if request.method == "POST":
        # Get the data from the POST request.
        exp = request.form['exp']
        exp = float(exp)
        # Make prediction using model loaded from disk as per the data.
        prediction = model.predict([[exp]])

        # Take the first value of prediction
        output = prediction[0][0]

        return "The approximate salary is: " + str(output)

if __name__ == '__main__':
   app.run(debug = True)
