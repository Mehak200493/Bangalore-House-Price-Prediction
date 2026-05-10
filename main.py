import pandas as pd
from flask import Flask, render_template, request
import pickle

# FIRST create app
app = Flask(__name__)

# Load data & model
data = pd.read_csv('Cleaned_data.csv')
pipe = pickle.load(open("RidgeModel.pkl", "rb"))

# Home route
@app.route('/')
def index():
    locations = sorted(data['location'].unique())
    return render_template('index.html', locations=locations)


# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    location = request.form.get('location')
    custom_location = request.form.get('custom_location')

    if location == "other":
        location = custom_location

    bhk = int(request.form.get('bhk'))
    bath = int(request.form.get('bath'))
    sqft = float(request.form.get('total_sqft'))

    if location not in data['location'].unique():
        location = 'Whitefield'   # fallback

    input_df = pd.DataFrame([[location, sqft, bath, bhk]],
                            columns=['location', 'total_sqft', 'bath', 'bhk'])

    prediction = pipe.predict(input_df)[0]

    if prediction >= 100:
        result = f"{round(prediction/100, 2)} Crores"
    else:
        result = f"{round(prediction, 2)} Lakhs"

    return result


# Run app (LAST)
if __name__ == "__main__":
    app.run(debug=True, port=5001)