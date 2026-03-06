import streamlit as st
from datetime import date
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# Load trained model
model = joblib.load("./model/decisiontree_classifier_baseline.pkl")

# we have added this as part of class assignment
nbcmodel = joblib.load("./model/naive_Bayes_classifier_optimum.pkl")
knnmodel = joblib.load("./model/knn_classifier_optimum.pkl")
rfcmodel = joblib.load("./model/random_forest_classifier_optimum.pkl")
svmmodel = joblib.load("./model/support_vector_classifier_optimum.pkl")
decisiontree_regressor_optimum = joblib.load('./model/decisiontree_regressor_optimum.pkl')
label_encoders_1b = joblib.load('./model/label_encoders_1b.pkl')

#confirm with Robbi about this
label_encoders_path = joblib.load('./model/scaler_5.pkl')
onehot_encoder_path = joblib.load('./model/onehot_encoder_3.pkl')


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
tab1, tab2, tab3, tab4 = st.tabs([
    "Customer Churn",
    "Predict Profit",
    "Predict Lateness ( KNN )",
    "Placeholder"
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
            options=["Business", "Individual"],
            index=None,
            placeholder="Choose an option...",
            )
        
        branch_sub_county = st.text_input("Branch Sub-County", "e.g., Kilimani")
        product_category_name = st.text_input("Product Category Name", "e.g., Meat-Based Dishes")
        quantity_ordered = st.number_input("Quantity Ordered")
        payment_date = st.date_input("Payment Date", date(2030, 7, 6))

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
        # Build the initial dictionary (mimicking your JSON request)
        data = {
            'PaymentDate': payment_date,
            'CustomerType': customer_type_selection,
            'BranchSubCounty': branch_sub_county,
            'ProductCategoryName': product_category_name,
            'QuantityOrdered': quantity_ordered
            }
        
        

        # Convert to DataFrame
        new_data = pd.DataFrame([data])

        # Feature Engineering (Date)
        # We can use the date attributes directly since 'payment_date' is already a date object
        new_data['PaymentDate_year'] = payment_date.year
        new_data['PaymentDate_month'] = payment_date.month
        new_data['PaymentDate_day'] = payment_date.day
        new_data['PaymentDate_dayofweek'] = payment_date.weekday()

        # Encode Categorical Columns
        # Note: Ensure 'label_encoders_1b' and your model are loaded in your script
        categorical_cols = ['CustomerType', 'BranchSubCounty', 'ProductCategoryName']
        for col in categorical_cols:
            new_data[col] = label_encoders_1b[col].transform(new_data[col])

        # Reorder to match training (expected_features)
        expected_features = [
            'CustomerType', 'BranchSubCounty', 'ProductCategoryName', 
            'QuantityOrdered', 'PaymentDate_year', 'PaymentDate_month', 
            'PaymentDate_day', 'PaymentDate_dayofweek'
        ]
        new_data = new_data[expected_features]

        # Predict
        prediction_regressor = decisiontree_regressor_optimum.predict(new_data)[0]

        # Output Result
        st.divider()

        st.write(col)
        st.write(new_data[col])
        st.write(label_encoders_1b[col].classes_)

        st.success(f"Prediction Profit: {prediction_regressor}")
        #st.subheader(f"Predicted Percentage Profit per Unit: {prediction_regressor:.2f}%")



# -----------------------------
# KNN
# -----------------------------
with tab3:

    st.header("Predict Lateness")

    with st.form("predict_lateness"):

        days_shipping_real = st.number_input("Days for shipping (real)")
        days_shipping_scheduled = st.number_input("Days for shipment (scheduled)")
        
        # 1. Define your mapping
        delivery_mapping = {
             "Yes": 1,
             "No": 0
             }
        # 2. Display the labels in the dropdown
        delivery_selection = st.selectbox(
            "Risk of Late Delivery",
            options=list(delivery_mapping.keys()), # This shows ["Late", "On Time"]
            index=None,
            placeholder="Choose an option..."
            )
        order_item_quantity = st.number_input("Order Item Quantity")

        sales = st.number_input("Sales")
        order_profit_per_order = st.number_input("Order Profit Per Order")
        
        shipping_mode = st.selectbox(
            "Shipping Mode",
            options=["Standard Class", "First Class"],
            index=None,
            placeholder="Choose an option...",
            )
        submit_lateness_prediction = st.form_submit_button("Predict Lateness")

    if submit_lateness_prediction:
        # # Build the initial dictionary (mimicking your JSON request)
        data = {
            'Days for shipping (real)': int(days_shipping_real),
            'Days for shipment (scheduled)': int(days_shipping_scheduled),
            'Late_delivery_risk':delivery_selection,
            'Order Item Quantity': int(order_item_quantity),
            'Sales': int(sales),
            'Order Profit Per Order': float(order_profit_per_order),
            'Shipping Mode': shipping_mode
            }
        
        

        # # Convert to DataFrame
        new_data = pd.DataFrame([data])

        scaler = StandardScaler()

        # One-hot encode 'Shipping Mode'
        encoded = onehot_encoder_path.transform(new_data[['Shipping Mode']])
        encoded_df = pd.DataFrame(encoded, columns=onehot_encoder_path.get_feature_names_out(['Shipping Mode']))
        new_data_preprocessed = pd.concat([new_data.drop('Shipping Mode', axis=1), encoded_df], axis=1)

        # Scale the features
        new_data_scaled = scaler.transform(new_data_preprocessed)

        # Make predictions
        predictionknn = knnmodel.predict(new_data_scaled)[0]

        

        st.write(col)
        st.write(new_data[col])
        st.write(label_encoders_1b[col].classes_)

        st.success(f"Prediction: {predictionknn}")
        # #st.subheader(f"Predicted Percentage Profit per Unit: {prediction_regressor:.2f}%")




# -----------------------------
# SALES MODEL FORM
# -----------------------------
with tab4:

    st.header("Sales Forecast")

    with st.form("sales_form"):

        marketing_spend = st.number_input("Marketing Spend")
        store_visits = st.number_input("Store Visits")

        submit_sales = st.form_submit_button("Predict Sales")

    if submit_sales:

        X = np.array([[marketing_spend, store_visits]])
        prediction = sales_model.predict(X)

        st.success(f"Predicted Sales: {prediction[0]}")