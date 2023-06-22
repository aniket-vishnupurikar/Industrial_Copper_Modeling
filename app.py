import pandas as pd
import numpy as np
from pickle import dump, load
from scipy.sparse import hstack
import streamlit as st
from streamlit_option_menu import option_menu
#df = pd.read_csv("copper cleaned data.csv")

st.markdown("<h1 style='text-align: center; color: yellow;'>Industrial Copper Modeling</h1>", unsafe_allow_html=True)

option = option_menu(None, ["App Info","Predict Selling Price","Predict Status"],
                       default_index=0, orientation="horizontal")
if option == "App Info":
    st.subheader("Problem Statement")
    st.write("The copper industry deals with less complex data related to sales and pricing. However, "
             "this data may suffer from issues such as skewness and noisy data, which can affect the accuracy of manual predictions."
             " Dealing with these challenges manually can be time-consuming and may not result in optimal pricing decisions. "
             "A machine learning regression model can address these issues by utilizing advanced techniques such as data "
             "normalization, feature scaling, and outlier detection, and leveraging algorithms that are robust to skewed and "
             "noisy data. Another area where the copper industry faces challenges is in capturing the leads. A lead classification "
             "model is a system for evaluating and classifying leads based on how likely they are to become a customer . "
             "You can use the STATUS variable with WON being considered as Success and LOST being considered as Failure and "
             "remove data points other than WON, LOST STATUS values.")
    st.subheader("App Introduction")
    st.write("There are two options to use this app. One option is to predict selling price given other variables and other is "
             "to predict status given other  variables.")
country_codes = (28.0, 25.0, 30.0, 32.0, 38.0, 78.0, 27.0, 77.0, 113.0, 79.0, 26.0, 39.0, 40.0, 84.0, 80.0, 107.0, 89.0)
status_codes = ('Won', 'Draft', 'To be approved', 'Lost', 'Not lost for AM', 'Wonderful', 'Revised', 'Offered', 'Offerable')
item_type_codes = ('W', 'WI', 'S', 'Others', 'PL', 'IPL', 'SLAWR')
application_codes = (10.0, 41.0, 28.0, 59.0, 15.0, 4.0, 38.0, 56.0, 42.0, 26.0, 27.0, 19.0, 20.0, 66.0, 29.0,
                     22.0, 40.0, 25.0, 67.0, 79.0, 3.0, 99.0, 2.0, 5.0, 39.0, 69.0, 70.0, 65.0, 58.0, 68.0)
#mat_ref_codes = tuple(df["material_ref"].unique())

if option == "Predict Selling Price":
    quantity_tons = st.number_input('Quantity Tons(should be greater than zero)', value=50)
    country = st.selectbox('Select Valid Country Code',country_codes)
    status = st.selectbox("Select Valid Status Code", status_codes)
    item_type = st.selectbox("Select Valid item Type Code", item_type_codes)
    application = st.selectbox("select application", application_codes)
    thickness = st.number_input("Thickness(should be greater than zero)", 1)
    width = st.number_input("Width(should be greater than zero)", 1)
    material_ref = st.text_input("Input Valid Material Reference No.", value="S0380700")
    product_ref = st.number_input("Input Valid Product Reference No.", value=1668701718)


    def load_and_transform(quantity_tons, country, status, item_type, application, thickness, width, material_ref,
                           product_ref):
        pt_qty = load(open('objects/pt_qty.pkl', 'rb'))
        quantity_tons_2 = pt_qty.transform(np.array(quantity_tons).reshape(-1, 1))[0][0]
        pt_thickness = load(open('objects/pt_thickness.pkl', 'rb'))
        thickness_2 = pt_thickness.transform(np.array(thickness).reshape(-1, 1))[0][0]
        pt_sp = load(open('objects/pt_sp.pkl', 'rb'))  ## to be used to inverse transform the final prediction
        ohe_country = load(open('objects/ohe_country.pkl', 'rb'))
        country_2 = ohe_country.transform(np.array(country).reshape(-1, 1))
        ohe_status = load(open('objects/ohe_status.pkl', 'rb'))
        status_2 = ohe_status.transform(np.array(status).reshape(-1, 1))
        ohe_item = load(open('objects/ohe_item.pkl', 'rb'))
        item_type_2 = ohe_item.transform(np.array(item_type).reshape(-1, 1))
        ohe_app = load(open('objects/ohe_app.pkl', 'rb'))
        application_2 = ohe_app.transform(np.array(application).reshape(-1, 1))
        ohe_mat = load(open('objects/ohe_mat.pkl', 'rb'))
        material_ref_2 = ohe_mat.transform(np.array(material_ref).reshape(-1, 1))
        ohe_pro = load(open('objects/ohe_pro.pkl', 'rb'))
        product_ref_2 = ohe_pro.transform(np.array(product_ref).reshape(-1, 1))
        X = hstack((np.array([width, quantity_tons_2, thickness_2]), country_2, status_2, item_type_2, application_2,
                    material_ref_2,
                    product_ref_2))
        return X


    X2 = load_and_transform(quantity_tons, country, status, item_type, application, thickness, width, material_ref,
                            product_ref)


    def predict(X):
        dt = load(open("objects/dt_model.pkl", "rb"))
        y_hat = dt.predict(X)[0]
        pt_sp = load(open('objects/pt_sp.pkl', 'rb'))
        sp = pt_sp.inverse_transform(np.array(y_hat).reshape(-1, 1))[0][0]
        return round(sp, 2)
    if st.button("Predict"):
        result = predict(X2)
        st.metric("Prediction", result)

if option == "Predict Status":
    quantity_tons = st.number_input('Quantity Tons(should be greater than zero)', value=50)
    country = st.selectbox('Select Valid Country Code', country_codes)
    selling_price = st.number_input("Selling Price(Should be greater than zero)", value=1000)
    item_type = st.selectbox("Select Valid item Type Code", item_type_codes)
    application = st.selectbox("select application", application_codes)
    thickness = st.number_input("Thickness(should be greater than zero)", 1)
    width = st.number_input("Width(should be greater than zero)", 1)
    material_ref = st.text_input("Input Valid Material Reference No.", value="S0380700")
    product_ref = st.number_input("Input Valid Product Reference No.", value=1668701718)


    def load_and_transform(quantity_tons, country, selling_price, item_type, application, thickness, width,
                           material_ref, product_ref):
        pt_qty = load(open('objects/pt_qty.pkl', 'rb'))
        quantity_tons_2 = pt_qty.transform(np.array(quantity_tons).reshape(-1, 1))[0][0]
        pt_thickness = load(open('objects/pt_thickness.pkl', 'rb'))
        thickness_2 = pt_thickness.transform(np.array(thickness).reshape(-1, 1))[0][0]
        pt_sp = load(open('objects/pt_sp.pkl', 'rb'))
        selling_price_2 = pt_sp.transform(np.array(selling_price).reshape(-1, 1))[0][0]
        ohe_country = load(open('objects/ohe_country2.pkl', 'rb'))
        country_2 = ohe_country.transform(np.array(country).reshape(-1, 1))
        ohe_item = load(open('objects/ohe_item2.pkl', 'rb'))
        item_type_2 = ohe_item.transform(np.array(item_type).reshape(-1, 1))
        ohe_app = load(open('objects/ohe_app2.pkl', 'rb'))
        application_2 = ohe_app.transform(np.array(application).reshape(-1, 1))
        ohe_mat = load(open('objects/ohe_mat2.pkl', 'rb'))
        material_ref_2 = ohe_mat.transform(np.array(material_ref).reshape(-1, 1))
        ohe_pro = load(open('objects/ohe_pro2.pkl', 'rb'))
        product_ref_2 = ohe_pro.transform(np.array(product_ref).reshape(-1, 1))
        X = hstack((np.array([width, quantity_tons_2, thickness_2]), country_2, selling_price_2, item_type_2,
                    application_2, material_ref_2,
                    product_ref_2))
        return X


    X2 = load_and_transform(quantity_tons, country, selling_price, item_type, application, thickness, width,material_ref,
                            product_ref)


    def predict(X):
        dt = load(open("objects/dt2.pkl", "rb"))
        y_hat = dt.predict(X)[0]
        le = load(open('objects/le.pkl', 'rb'))
        status = le.inverse_transform(np.array(y_hat).reshape(-1, 1))[0]
        return status

    if st.button("Predict"):
        result = predict(X2)
        st.metric("Prediction", result)










