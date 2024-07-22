

# Simple Iris Prediction App

This is a simple web application built using Streamlit that predicts the type of Iris flower based on user input parameters. The app utilizes a RandomForestClassifier from scikit-learn to make predictions.

## Features

- **Interactive Sliders:** Input sepal length, sepal width, petal length, and petal width using sliders.
- **User Input Display:** Shows the user input parameters.
- **Flower Type Prediction:** Predicts the Iris flower type based on the input parameters.
- **Prediction Probability:** Displays the prediction probabilities for each class.

## Prerequisites

Ensure you have the following packages installed:

- pandas
- matplotlib
- seaborn
- scikit-learn
- streamlit

You can install the required packages using pip:

```sh
pip install pandas matplotlib seaborn scikit-learn streamlit
```

## How to Run the App

1. Save the provided code into a file, for example, `iris_app.py`.
2. Open a terminal and navigate to the directory where the file is saved.
3. Run the Streamlit app using the following command:

```sh
streamlit run iris_app.py
```

4. A new tab will open in your default web browser with the app running.

## Code Explanation

### Import Libraries

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
import streamlit as st
```

### Title and Description

```python
st.write("""
# Simple Iris Prediction App

This app predicts the **Iris Flower** type!!
""")
```

### Sidebar for User Input

```python
st.sidebar.header("User Input Parameters")

def user_input_features():
    sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('Sepal Width', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 0.2)

    data = {
        'sepal_length': sepal_length,
        'sepal_width': sepal_width,
        'petal_length': petal_length,
        'petal_width': petal_width
    }

    t_data = pd.DataFrame(data, index=[0])
    return t_data

x_test = user_input_features()
```

### Display User Input Parameters

```python
st.subheader('User Input Parameters')
st.write(x_test)
```

### Load Dataset and Train Model

```python
df = datasets.load_iris()
x_train = df.data
y_train = df.target

forest_classifier = RandomForestClassifier()
forest_classifier.fit(x_train, y_train)
```

### Make Predictions

```python
predict = forest_classifier.predict(x_test)
prediction_prob = forest_classifier.predict_proba(x_test)
```

### Display Results

```python
st.subheader('Class Labels and Their Corresponding Index Number')
st.write(df.target_names)

st.subheader('Prediction')
st.write(df.target_names[predict])

st.subheader('Prediction Probability')
st.write(prediction_prob)
```

## Conclusion

This simple web application demonstrates how to use Streamlit to build an interactive machine learning model. The app predicts the type of Iris flower based on user input parameters using a RandomForestClassifier.

Feel free to customize and expand this app as needed!

---
.
