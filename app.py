from flask import Flask, render_template, request, redirect, url_for, session
import pickle
import pandas as pd

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Needed for session

# Load model and encoders
model = pickle.load(open("model.pkl", "rb"))
encoders = pickle.load(open("encoders.pkl", "rb"))

@app.route('/')
def home():
    prediction = session.pop('prediction_text', None)  # remove after showing
    return render_template("index.html", prediction_text=prediction)

@app.route("/predict", methods=["POST"])
def predict():
    data = {
        'A1_Score': int(request.form['A1_Score']),
        'A2_Score': int(request.form['A2_Score']),
        'A3_Score': int(request.form['A3_Score']),
        'A4_Score': int(request.form['A4_Score']),
        'A5_Score': int(request.form['A5_Score']),
        'A6_Score': int(request.form['A6_Score']),
        'A7_Score': int(request.form['A7_Score']),
        'A8_Score': int(request.form['A8_Score']),
        'A9_Score': int(request.form['A9_Score']),
        'A10_Score': int(request.form['A10_Score']),
        'age': int(request.form['age']),
        'gender': request.form['gender'],
        'ethnicity': request.form['ethnicity'],
        'jaundice': request.form['jaundice'],
        'austim': request.form['austim'],
        'contry_of_res': request.form['contry_of_res'],
        'used_app_before': request.form['used_app_before'],
        'result': float(request.form['result']),
        'relation': request.form['relation']
    }

    df = pd.DataFrame([data])

    # Encode categorical columns
    categorical_columns = ['gender', 'ethnicity', 'jaundice', 'austim', 'contry_of_res', 'used_app_before', 'relation']
    for col in categorical_columns:
        if col in encoders:
            df[col] = encoders[col].transform(df[col])

    prediction = model.predict(df)[0]
    result = "ASD Positive" if prediction == 1 else "ASD Negative"
    session['prediction_text'] = f"Prediction: {result}"
    
    return redirect(url_for("home"))

if __name__ == '__main__':
    app.run(debug=True)
