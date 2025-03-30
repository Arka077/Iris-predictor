import streamlit as st
import pandas as pd
import numpy as np
from joblib import load

model = load('C:/Users/ASUS/coding/projects/iris_prediction_model/model.joblib')

st.title('Iris Species Predictor')

st.subheader('Enter the details:')

sepal_length=st.number_input('Sepal length(cm)')
sepal_width=st.number_input('Sepal width(cm)')
petal_length=st.number_input('Petal length(cm)')
petal_width=st.number_input('Petal width(cm)')

map={
    (1,0,0):'Setosa',
    (0,1,0):'Versicolor',
    (0,0,1):'Virginica'
}

if st.button('Predict species'):
    input_data=pd.DataFrame([[sepal_length,sepal_width,petal_length,petal_width]],
                            columns=['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)'])

    prediction=model.predict(input_data)[0]

    pred_tuple=tuple(prediction)

    species=map.get(pred_tuple,'Unknown')

    st.write(f'Predicted species:{species}')

# streamlit run c:\Users\ASUS\coding\projects\iris_prediction_model\app.py 