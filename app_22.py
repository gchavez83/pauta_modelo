import pandas as pd
import streamlit as st
#import numpy as np
from pycaret.regression import load_model, predict_model

st.set_page_config(page_title="Modelo de Predicción de Resultados de Anuncios")

@st.cache(allow_output_mutation=True)
def get_model():
    return load_model("final_rf_pauta")

def predict(model, df):
    predictions = predict_model(model, data = df)
    return predictions['Label'][0]

model = get_model()

st.title("Modelo de Predicción de Resultados de Anuncios")
st.markdown("Elija los valores para pronosticar el Resultado del Anuncio)")

form = st.form("anuncios")
Importe_invertido = form.number_input('Importe Invertido', min_value = 0.0 , max_value = 20000.00,value=2000.00 , format = '%.2f', step = 1)
clasificacion_list = ['Atributos','Especiales','Estratégicas','Estratégico','Spots','Territorio']
Clasificacion_Descripcion = form.selectbox('Clasificacion', clasificacion_list)
objetivo_list = ['Interacción','ThruPlay']
Objetivo_Descripcion = form.selectbox('Objetivo', objetivo_list)
redsocial_list = ['Dark Post', 'Facebook','Instagram']
RedSocial_Descripcion = form.selectbox('Red Social', redsocial_list)
#Cuadrante_list = ['NE', 'NO', 'SE', 'SO']
#Cuadrante = form.selectbox('Cuadrante', Cuadrante_list)
#Distance_Km = form.slider('Distance Km', min_value = 0.0 , max_value = 15.0 , value = 2.0, format = '%.2f' )

predict_button = form.form_submit_button('Predict')

input_dict = {'Importe gastado (MXN)': Importe_invertido, 'Clasificacion.Descripción': Clasificacion_Descripcion, 'Objetivo.Descripción': Objetivo_Descripcion, 'Red_Social.Descripción': RedSocial_Descripcion}

input_df = pd.DataFrame([input_dict])

if predict_button: 
    out = predict(model, input_df)
    st.success(f'La prediccion del predio es {out}.')