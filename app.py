import pandas as pd 
import numpy as np 
import pickle as pk 
import streamlit as st

model = pk.load(open('ocarmodel.pkl', 'rb'))

st.header('Car Price Prediction ML Model')

# Load car dataset
cars_data = pd.read_csv(r'C:\Users\PRAJWAL\OneDrive\Desktop\project\realcar.csv')

# Extract brand name from car name
def get_brand_name(car_name):
    return car_name.split(' ')[0].strip()

cars_data['name'] = cars_data['name'].apply(get_brand_name)

# UI Elements
name = st.selectbox('Select Car Brand', sorted(cars_data['name'].unique()))
year = st.slider('Car Manufactured Year', 1994, 2024)
km_driven = st.slider('No of kms Driven', 11, 200000)
options=["4","5","6","7","8","9","10","14"]
seats = st.selectbox('No of Seats',options)
fuel = st.selectbox('Fuel type', cars_data['fuel'].unique())
seller_type = st.selectbox('Seller type', cars_data['seller_type'].unique())
transmission = st.selectbox('Transmission type', cars_data['transmission'].unique())
owner = st.selectbox('Owner type', cars_data['owner'].unique())
mileage = st.slider('Car Mileage (kmpl)', 10, 40)
engine = st.slider('Engine CC', 700, 5000)
max_power = st.slider('Max Power (bhp)', 0, 300)


if st.button("Predict"):
    # Prepare input data
    input_data_model = pd.DataFrame(
        [[name, year, km_driven, fuel, seller_type, transmission, owner, mileage, engine, max_power, seats]],
        columns=['name', 'year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner', 'mileage', 'engine', 'max_power', 'seats']
    )

    # Encode categorical variables to match training format
    input_data_model['owner'].replace(
        ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner', 'Test Drive Car'],
        [1, 2, 3, 4, 5], inplace=True)

    input_data_model['fuel'].replace(['Diesel', 'Petrol', 'LPG', 'CNG'], [1, 2, 3, 4], inplace=True)

    input_data_model['seller_type'].replace(['Individual', 'Dealer', 'Trustmark Dealer'], [1, 2, 3], inplace=True)

    input_data_model['transmission'].replace(['Manual', 'Automatic'], [1, 2], inplace=True)

    input_data_model['name'].replace(
        ['Maruti', 'Skoda', 'Honda', 'Hyundai', 'Toyota', 'Ford', 'Renault',
         'Mahindra', 'Tata', 'Chevrolet', 'Datsun', 'Jeep', 'Mercedes-Benz',
         'Mitsubishi', 'Audi', 'Volkswagen', 'BMW', 'Nissan', 'Lexus',
         'Jaguar', 'Land', 'MG', 'Volvo', 'Daewoo', 'Kia', 'Fiat', 'Force',
         'Ambassador', 'Ashok', 'Isuzu', 'Opel'],
        list(range(1, 32)), inplace=True)

    # Prediction
    car_price = model.predict(input_data_model)

    # Display result
    st.success(f"Predicted Car Price: ₹ {car_price[0]:,.2f}")
