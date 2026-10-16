import pickle

import streamlit as st
import pandas as pd



#load the model (model deserialization)
with open("best_model/best_model_pipeline.pkl","rb") as file:
    best_pipeline = pickle.load(file)

st.set_page_config(page_title="Loan Approval Classification",page_icon="assets/AI Loan Approval Icon.png")

with st.sidebar:
    st.title("Loan Approval Classification")
    st.image("assets/AI Loan Approval Icon.png")

#load the dataset
df = pd.read_csv("dataset/cleaned_loan_data.csv")


col1,col2,col3 = st.columns(3,border=True)

with col1:
    person_age= st.slider("Age",min_value=1,max_value=100)
    person_gender=st.radio("Gender", ["Female", "Male"], horizontal=True)
    person_education=st.selectbox("Education",options=df['person_education'].unique())	
    person_income=st.number_input("Income",min_value=10000.0,max_value=1000000.0,step=1000.0
                                  )	
    

with col2:
    person_emp_exp=st.number_input("Employee Experience",min_value=0,max_value=10,step=1)	
    person_home_ownership=st.selectbox('Home Ownership',options=df['person_home_ownership'].unique())
    loan_amnt=st.number_input("Loan Amount",min_value=1000.0,max_value=1000000.0,step=1000.0)		
    loan_intent	= st.selectbox('Loan Type',options=df['loan_intent'].unique())
    

with col3:    
    loan_int_rate=st.number_input("Loan Int Rate",min_value=1.0,max_value=100.0,step=2.0)	
    loan_percent_income	= st.number_input("Loan Percent Income",min_value=0.0,max_value=100.0,step=1.0)
    cb_person_cred_hist_length	= st.number_input("Cred Hist Length",min_value=0.0,max_value=50.0,step=1.0)
    credit_score=st.number_input("Credit Score",min_value=1.0,max_value=10000.0,step=100.0)
    previous_loan_defaults_on_file=st.selectbox('Previous Loan Pay Or Not',options=df['previous_loan_defaults_on_file'].unique())


st.header("🔍Prediction Result")
if  st.button("Check Loan Approval"):
    missing_values = [
        person_age,person_gender,person_education,person_income,person_emp_exp,person_home_ownership,
        loan_amnt,loan_intent,loan_int_rate,loan_percent_income,cb_person_cred_hist_length,
        credit_score,previous_loan_defaults_on_file
    ]
    if any(value is None for value in missing_values):
        st.error("Please fill in all the required fields.")
    else:
        input_data = pd.DataFrame({
            'person_age': [person_age],
            'person_gender': [person_gender.lower()],
            'person_education': [person_education],
            'person_income': [person_income],
            'person_emp_exp': [person_emp_exp],
            'person_home_ownership': [person_home_ownership],
            'loan_amnt': [loan_amnt],
            'loan_intent': [loan_intent],
            'loan_int_rate': [loan_int_rate],
            'loan_percent_income': [loan_percent_income],
            'cb_person_cred_hist_length': [cb_person_cred_hist_length],
            'credit_score': [credit_score],
            'previous_loan_defaults_on_file': [previous_loan_defaults_on_file]
        })

        prediction = best_pipeline.predict(input_data)
        if prediction[0] == 1:
            st.success("Congratulations! Your loan is likely to be approved.")
        else:
            st.error("Unfortunately, your loan is likely to be denied.")