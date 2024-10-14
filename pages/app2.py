# pages/app2.py

import os
import streamlit as st
import plotly.express as px
from helpers import predictor

def app():
    st.write("# Project garbAIge")

    st.write("## Upload Image in .jpg format")
    uploaded_image = st.file_uploader("", type=["jpg"])

    st.write("## Uploaded Image")

    if uploaded_image:
        st.image(uploaded_image)

        button = st.button("Classify", key=None)

        if button:
            # Save the uploaded image temporarily
            with open("temp.jpg", "wb") as f:
                f.write(uploaded_image.getbuffer())
            
            prediction, predicted_class = predictor.predict("temp.jpg")

            if prediction is not None:
                labels = {0: 'cardboard', 1: 'glass', 2: 'metal', 3: 'paper', 4: 'plastic', 5: 'trash'}

                classes = []
                prob = []
                for i, j in enumerate(prediction[0], 0):
                    classes.append(labels[i].capitalize())
                    prob.append(round(j * 100, 2))

                fig = px.bar(x=classes, y=prob,
                             text=prob, color=classes,
                             labels={"x": "Material", "y": "Probability(%)"})

                st.markdown("#### Probability Distribution Bar Chart")
                st.plotly_chart(fig)

                st.markdown(f"#### The Image Is Classified As `{predicted_class.capitalize()}` With A Probability Of `{max(prob)}%`", True)
            else:
                st.error("Prediction Failed. Please try again.")
                
    else:
        st.write("#### No Image Was Found, Please Retry!!!")
