import streamlit as st
import pandas as pd
import pickle
from custom_transformers import binary_label_encoder, gender_encoder
# Load the model
model = pickle.load(open('models/Best_LogisticRegression_Model.pkl', 'rb'))

pipe=pickle.load(open('models/pipe.pkl', 'rb'))

pipe_2=pickle.load(open('models/pipeline_segmentation.pkl', 'rb'))
def main():
    st.title("Churn Prediction Dashboard")

    # User input
    st.sidebar.header("Enter the Information of User's")


    gender=st.sidebar.selectbox("Gender",['Male','Female'])
    SeniorCitizen=st.sidebar.selectbox("Senior Citizen",['Yes','No'])
    
    Partner=st.sidebar.selectbox("Partner",['Yes','No'])
    Dependents=st.sidebar.selectbox("Dependents",['Yes','No'])
    tenure=st.sidebar.slider("Tenure",0,100,step=1)
    PhoneService=st.sidebar.selectbox("Phone Service",['Yes','No'])
    MultipleLines=st.sidebar.selectbox("Multiple Lines",['Yes','No','No phone service'])
    InternetService=st.sidebar.selectbox("Internet Service",['DSL','Fiber optic','No'])
    OnlineSecurity=st.sidebar.selectbox("Online Security",['Yes','No','No internet service'])
    OnlineBackup=st.sidebar.selectbox("Online Backup",['Yes','No','No internet service'])
    DeviceProtection=st.sidebar.selectbox("Device Protection",['Yes','No','No internet service'])
    TechSupport=st.sidebar.selectbox("Tech Support",['Yes','No','No internet service'])
    StreamingTV=st.sidebar.selectbox("Streaming TV",['Yes','No','No internet service'])
    StreamingMovies=st.sidebar.selectbox("Streaming Movies",['Yes','No','No internet service'])
    Contract=st.sidebar.selectbox("Contract",['Month-to-month','One year','Two year'])
    PaperlessBilling=st.sidebar.selectbox("Paperless Billing",['Yes','No'])
    PaymentMethod=st.sidebar.selectbox("Payment Method",['Electronic check','Mailed check','Bank transfer (automatic)','Credit card (automatic)'])
    MonthlyCharges=st.sidebar.slider("Monthly Charges",0,120,step=1)
    # Calculate TotalCharges
    TotalCharges = tenure * MonthlyCharges


    # Process the input data
    input_data = {
        'gender': [gender],
        'SeniorCitizen': [SeniorCitizen],
        'Partner': [Partner],
        'Dependents': [Dependents],
        'tenure': [tenure],
        'PhoneService': [PhoneService],
        'MultipleLines': [MultipleLines],
        'InternetService': [InternetService],
        'OnlineSecurity': [OnlineSecurity],
        'OnlineBackup': [OnlineBackup],
        'DeviceProtection': [DeviceProtection],
        'TechSupport': [TechSupport],
        'StreamingTV': [StreamingTV],
        'StreamingMovies': [StreamingMovies],
        'Contract': [Contract],
        'PaperlessBilling': [PaperlessBilling],
        'PaymentMethod': [PaymentMethod],
        'MonthlyCharges': [MonthlyCharges],
        'TotalCharges': [TotalCharges]
    }
    input_data_seg={
        'gender':[gender],
        'SeniorCitizen':[SeniorCitizen],
        'Partner':[Partner],
        'tenure':[tenure],
        'PhoneService':[PhoneService],
        'Contract':[Contract],
        'PaperlessBilling':[PaperlessBilling],
        'PaymentMethod':[PaymentMethod],
        'MonthlyCharges':[MonthlyCharges],
        'TotalCharges':[TotalCharges]
    }
        
# SeniorCitizen          2
# Partner                2
# tenure                73
# PhoneService           2
# Contract               3
# PaperlessBilling       2
# PaymentMethod          4
# MonthlyCharges      1585
# TotalCharges        6530
# Churn 
    
    
    # Convert the data to a dataframe
    input_df = pd.DataFrame(input_data)

    # Encoding
    transformed_ip=pipe.transform(input_df)
    # now for the fitting in pipe we need to convert the data frame to numpy array.
    
    if st.button("Predict"):
        prediction = model.predict(transformed_ip)
        if prediction[0]==1:
            st.write("Customer will Churn")
        else:
            st.write("Customer will not Churn")
    

if __name__=='__main__':
    main()


