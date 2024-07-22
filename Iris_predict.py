import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
import streamlit as st


st.write("""
# Simple Iris Prediction App

 This app predicts the **Iris Flower** type!!
""")

st.sidebar.header("User Input Parameters ")

def user_input_features():
    sepal_length = st.sidebar.slider('Sepal lenght',4.3,7.9,5.4)
    sepal_width = st.sidebar.slider('Sepal Width',2.0,4.4,3.4)
    petal_length = st.sidebar.slider('Petal lenght',1.0,6.9,1.3)
    petal_width = st.sidebar.slider('Sepal lenght',0.1,2.5,0.2)

    data = {
        'sepal_length':sepal_length,
        'sepal_width':sepal_width,
        'petal_length':petal_length,
        'petal_width':petal_width
    }

    t_data = pd.DataFrame(data, index= [0])
    return t_data



x_test = user_input_features()

st.subheader('User Input Parameter')
st.write(x_test)

# lets load dataset
df = datasets.load_iris()
x_train = df.data
y_train = df.target
print(x_train)

forest_classifier = RandomForestClassifier()
forest_classifier.fit(x_train,y_train)

predict = forest_classifier.predict(x_test)
prediction_prob = forest_classifier.predict_proba(x_test)

st.subheader('Class labels and their corresponding index number')
st.write(df.target_names,)

st.subheader('Prediction')
st.write(df.target_names[predict])

st.subheader('Prediction Probability')
st.write(prediction_prob)

