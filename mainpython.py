import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load Titanic dataset
@st.cache
def load_data():
    #url = "https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv"
    data = pd.read_csv('Dataset/train.csv')
    return data

# Sidebar for user input
st.sidebar.title("Titanic Explorer")
st.sidebar.subheader("Choose Your Adventure")

# Load data
data = load_data()

# Show dataset
if st.sidebar.checkbox("Show Dataset"):
    st.subheader("Titanic Dataset")
    st.write(data)

# Data Exploration
if st.sidebar.checkbox("Explore Data"):
    st.subheader("Explore Data")
    st.write("Number of Survived and Not Survived Passengers:")
    st.write(data['Survived'].value_counts())

# Machine Learning Model
if st.sidebar.checkbox("Train ML Model"):
    st.subheader("Train Machine Learning Model")
    # Prepare data for model training
    X = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
    y = data['Survived']
    X['Sex'] = X['Sex'].map({'male': 0, 'female': 1})
    X = X.fillna(X.mean())

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a simple Random Forest Classifier
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    st.write("Model Accuracy:", accuracy)

# Footer
st.sidebar.text("Built with Streamlit by Josmy Mathew")