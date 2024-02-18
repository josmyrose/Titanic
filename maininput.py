import streamlit as st
import pandas as pd
#from sklearn.externals import joblib  # For loading the pre-trained model
#import sklearn.external.joblib as extjoblib
import joblib
# Load the pre-trained model
model = joblib.load('model.pkl')  # Replace 'your_model_file.pkl' with your actual model file

# Streamlit app
def main():
    st.title("Titanic Survival Prediction App")

    # Collect user inputs
    pclass = st.selectbox("Select Pclass", [1, 2, 3])
    sex = st.selectbox("Select Sex", ["male", "female"])
    age = st.number_input("Enter Age", value=30.0, min_value=0.0, max_value=100.0)
    sibsp = st.number_input("Enter Siblings/Spouses Aboard", value=0, min_value=0, max_value=10)
    parch = st.number_input("Enter Parents/Children Aboard", value=0, min_value=0, max_value=10)
    fare = st.number_input("Enter Fare", value=20.0, min_value=0.0, max_value=500.0)
    
    embarked_options = ["C", "Q", "S"]
    embarked = st.selectbox("Select Embarked", embarked_options)

    # Make prediction on button click
    if st.button("Predict"):
        # Create a DataFrame with user inputs
        input_data = pd.DataFrame({
            'Pclass': [pclass],
            'Sex': [sex],
            'Age': [age],
            'SibSp': [sibsp],
            'Parch': [parch],
            'Fare': [fare],
            'Embarked': [embarked]
        })

        # Convert categorical variables to dummy variables
        input_data = pd.get_dummies(input_data, columns=['Sex', 'Embarked'])

        # Make prediction
        prediction = model.predict(input_data)

        # Display prediction result
        if prediction[0] == 1:
            st.success("Survived")
        else:
            st.error("Did not survive")

if __name__ == "__main__":
    main()