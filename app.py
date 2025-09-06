from flask import Flask, request, render_template
from predict.predict import predict_new_house

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        try:
            # Get user input from form
            med_inc = float(request.form['med_inc'])
            house_age = float(request.form['house_age'])
            # Predict using the existing function
            prediction = predict_new_house(med_inc=med_inc, house_age=house_age)
        except Exception as e:
            prediction = f"Error: {str(e)}"
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
