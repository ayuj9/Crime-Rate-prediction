import numpy as np
from flask import Flask, render_template, request
import pickle
app = Flask(__name__)
preprocessor = pickle.load(open('preprocessor.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))


@app.route("/")
def home():
    return render_template('index.html')


@app.route('/prediction', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        state = request.form.get('State')
        year = request.form.get('Year')
        test_input = np.array([year, state], dtype=object).reshape(1, 2)
        test_input_state = preprocessor.transform(
        test_input[:, 1].reshape(1, 1))
        test_input_rem = test_input[:, 0].reshape(1, 1)
        test_input_transformed = np.concatenate((test_input_rem, test_input_state), axis=1)
        result = model.predict(test_input_transformed)
        return render_template('home.html', result=result[0])
    else:
        return render_template('home.html')


if __name__ == "__main__":
    app.run(host = '0.0.0.0')
