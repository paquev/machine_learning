import pandas as pd 
import numpy as np 
import joblib
import math 

from flask import Flask, jsonfly,render_template,request

app = Flask(__name__,template_folder='../template',static_folder='../template/css')

#cargar modelo
model = joblib.load('../models/diabetes_model.pkl')

@app.route('/',methods=['GET','POST'])
def prediction():
    if request.method == 'POST':

        #crear un dataframe 
        data = {
            'embarazos' : float(request.form.get('embarazos')),
            'glucosa' : float(request.form.get('glucosa')),
            'presion_sangre' : float(request.form.get('presion_sangre')),
            'gro_piel' : float(request.form.get('gro_piel')),
            'insulina' : float(request.form.get('insulina')),
            'grasa_corp' : float(request.form.get('grasa_corp')),
            'historial_f' : float(request.form.get('hitorico')),
            'edad' : float(request.form.get('edad'))
        }

        df_to_predict = pd.DataFrame(data=data)

        predict_model = round(model.predict(df_to_predict))
    else:
        predict_model = "ERROR!"

    return render_template('index.html',output=predict_model)

app.run(port=8000)