import streamlit as st
import joblib
import numpy as np

model = joblib.load('best_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("Heart Disease Prediction App")

age = st.number_input("Age", min_value=1, max_value=100)


if st.button("Predict Here"):
    input_data = np.array([[age]])
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)
    
    if prediction[0] == 1:
        st.write("হার্ট ডিজিজের ঝুঁকি রয়েছে।")
    else:
        st.write("হার্ট ডিজিজের ঝুঁকি নেই।")