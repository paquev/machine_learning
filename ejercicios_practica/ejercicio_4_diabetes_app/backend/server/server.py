import pandas as pd
import numpy as np
import joblib
import math
import os
import logging

from flask import Flask, render_template, request

# Initialize the app
app = Flask(__name__, template_folder='../../fronted/src/', static_folder='../../fronted/src/css/')

# Enable debugging logs
logging.basicConfig(level=logging.DEBUG)

# Load model
model = joblib.load('../models/diabetes_model.pkl')

@app.route('/', methods=['GET', 'POST'])
def prediction():
    if request.method == 'POST':
        try:
            # Create a dataframe with the data from the form
            data = {
                'embarazos': float(request.form.get('embarazos')),
                'glucosa': float(request.form.get('glucosa')),
                'presion_sangre': float(request.form.get('presionSangre')),
                'gro_piel': float(request.form.get('grosorPiel')),
                'insulina': float(request.form.get('insulina')),
                'imc': float(request.form.get('imc')),
                'historial_f': float(request.form.get('historialFamiliar')),
                'edad': float(request.form.get('edad'))
            }

            # Convert the data to a dataframe
            df_to_predict = pd.DataFrame(data=data, index=[0])
            X = np.array(df_to_predict)

            # Make the prediction
            prediction = model.predict(X)[0]

            # Determine if diabetic or not
            if prediction == 1:
                result = 'Diabetico'
                print(result)
            else:
                result = 'No diabetico'
                print(result)

            return render_template('index.html', output=result)

        except Exception as e:
            print(f"Error during prediction: {e}")
            return render_template('index.html', output="Error during prediction!")

    else:
        return render_template('index.html', output="Awaiting submission")


if __name__ == '__main__':
    # Print the path of the template folder for debugging purposes
    print("Template folder absolute path:", os.path.abspath(app.template_folder))
    app.run(port=8000, debug=True)
