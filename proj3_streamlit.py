import streamlit as st
import pandas as pd
import pickle


# Load the model from the pickle file
with open('random_forest_model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

# Load the CSV file containing the cleaned data
final_df = pd.read_csv("final_clean_csv.csv")

# Get unique values from the DataFrame for dropdown selections
f = list(final_df['ft'].unique())
b = list(final_df['bt'].unique())
t = list(final_df['transmission'].unique())
c = list(final_df['city'].unique())
s = list(final_df['Seats'].unique())

# Streamlit app title
st.title("Car Price Predicton")

# Create a form for user inputs
form = st.form(key='registration_form')

ft = form.selectbox("FUEL TYPE", f)
bt = form.selectbox("BODY TYPE", b)
tr = form.selectbox("TRANSMISSION", t)
ci = form.selectbox("CITY", c)
se = form.selectbox("NUMBER OF SEATS", s)
mil = form.slider("Mileage", min_value=7, max_value=35)

# Submit button for the form
submit_button = form.form_submit_button(label='Predict Price')

# When the submit button is pressed
if submit_button:
    # Prepare the input data as a DataFrame
    input_data = {
        'ft': [ft],
        'bt': [bt],
        'transmission': [tr],
        'city': [ci],
        'Seats': [se],
        'Mileage': [mil]
    }

    # Create a DataFrame from the input data
    input_df = pd.DataFrame(input_data)

    # Perform one-hot encoding for the categorical variables
    input_df_encoded = pd.get_dummies(input_df, drop_first=True)

    # Align the input_df_encoded with the model's expected feature columns
    # Get the list of columns used in training
    model_columns = loaded_model.feature_names_in_  # Correctly retrieving feature names

    # Ensure the DataFrame aligns with the model's features
    input_df_encoded = input_df_encoded.reindex(columns=model_columns, fill_value=0)

    # Make prediction
    prediction = loaded_model.predict(input_df_encoded)

    # Display the predicted price
    st.subheader(f"The predicted price of the car is: {prediction[0]:,.2f}LAKHS")   
