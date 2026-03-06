import streamlit as st
from datetime import date
import joblib
import numpy as np


# Load trained model
model = joblib.load("./model/decisiontree_classifier_baseline.pkl")

# we have added this as part of class assignment
nbcmodel = joblib.load("./model/naive_Bayes_classifier_optimum.pkl")
knnmodel = joblib.load("./model/knn_classifier_optimum.pkl")
rfcmodel = joblib.load("./model/random_forest_classifier_optimum.pkl")
svmmodel = joblib.load("./model/support_vector_classifier_optimum.pkl")
decisiontree_regressor_optimum = joblib.load('./model/decisiontree_regressor_optimum.pkl')



# Streamlit page config


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Group Work CAT",
    page_icon="📊",
    layout="wide"
)

st.title("Dashboard")

# -----------------------------
# Tabs for different models
# -----------------------------
tab1, tab2, tab3 = st.tabs([
    "Customer Churn",
    "Predict Profit",
    "Sales Forecast"
])

# -----------------------------
# CHURN MODEL FORM
# -----------------------------
with tab1:

    st.header("Customer Churn Prediction")

    with st.form("churn_form"):

        monthly_fee = st.number_input("Monthly Fee", min_value=0.0)
        age = st.number_input("Customer Age", min_value=0)
        support_calls = st.number_input("Support Calls", min_value=0)

        submit_churn = st.form_submit_button("Predict Churn")

    if submit_churn:

        X = np.array([[monthly_fee, age, support_calls]])
        prediction = model.predict(X)

        st.success(f"Churn Prediction: {prediction[0]}")


# -----------------------------
# Predict Profit
# -----------------------------
with tab2:

    st.header("Predict Profit")

    with st.form("predict_profit"):
        
        customer_type_selection = st.selectbox(
            "Select Customer Type",
            options=["Business", "Consumer", "Corporate"],
            index=None,
            placeholder="Choose an option...",
            )
        
        branch_sub_county = st.text_input("Branch Sub-County", "e.g., Kilimani")
        product_category_name = st.text_input("Product Category Name", "e.g., Meat-Based Dishes")
        quantity_ordered = st.number_input("Quantity Ordered")
        payment_date = st.date_input("Payment Date", date(2000, 7, 6))

        # 2. Extract Year, Month, and Day
        year_of_payment = payment_date.year
        month_of_payment = payment_date.month
        day_of_payment = payment_date.day

        # 3. Get the Day of the Week
        # .strftime("%A") returns the full name (e.g., "Thursday")
        day_name = payment_date.strftime("%A")

        # .weekday() returns an integer (0 for Monday, 6 for Sunday)
        day_index = payment_date.weekday()


        #transactions_today = st.number_input("Transactions Today")

        submit_profit_prediction = st.form_submit_button("Predict Profit")

    if submit_profit_prediction:

        'CustomerType',
        'BranchSubCounty',
        'ProductCategoryName',
        'QuantityOrdered',
        'PaymentDate_year',
        'PaymentDate_month',
        'PaymentDate_day',
        'PaymentDate_dayofweek'

        X = np.array([[customer_type_selection, branch_sub_county, product_category_name,quantity_ordered,year_of_payment,month_of_payment, day_of_payment,day_name ]])
        prediction = decisiontree_regressor_optimum.predict(X)

        st.success(f"Profit Prediction: {prediction[0]}")


# -----------------------------
# SALES MODEL FORM
# -----------------------------
with tab3:

    st.header("Sales Forecast")

    with st.form("sales_form"):

        marketing_spend = st.number_input("Marketing Spend")
        store_visits = st.number_input("Store Visits")

        submit_sales = st.form_submit_button("Predict Sales")

    if submit_sales:

        X = np.array([[marketing_spend, store_visits]])
        prediction = sales_model.predict(X)

        st.success(f"Predicted Sales: {prediction[0]}")