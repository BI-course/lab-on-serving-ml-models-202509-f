import streamlit as st
from datetime import date
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# Load trained model
model = joblib.load("./model/decisiontree_classifier_baseline.pkl")

# we have added this as part of class assignment
svmmodel = joblib.load("./model/support_vector_classifier_optimum.pkl")
decisiontree_regressor_optimum = joblib.load('./model/decisiontree_regressor_optimum.pkl')
label_encoders_1b = joblib.load('./model/label_encoders_1b.pkl')

#load rules
recommender_rules = pd.read_csv("./rules/top_rules_7b.csv")


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

st.header("Chosen Customer Churn, Predict Profit & Recommender")

# -----------------------------
# Tabs for different models
# -----------------------------
tab1, tab2, tab3 = st.tabs([
    "Customer Churn",
    "Predict Profit",
    "Recommender Rules"
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
        
        
        #branch_sub_county = st.text_input("Branch Sub-County", "e.g., Kilimani")
        branch_sub_county = st.selectbox(
            "Select Sub County",
            options=["Makadara","Kamukunji","Roy Sambu","Kibra","Langata","Kasarani","Mathare","Dagoretti","Starehe","Githurai","Nyeri Central","Kisumu Central","Nakuru Town East","Kesses","Embakasi","Westlands","Kilimani","Nyali","Kangemi","Ruaraka"],
            index=None,
            placeholder="Choose an option...",
            )
        
        product_category_name = st.selectbox(
            "Select Product Cateogry",
            options=["African Cultural Specials","Combination Plates","Fish Dishes","Fried Dishes","Legume-Based Dishes","Meat-Based Dishes","Rice Dishes","Soup/Stew Dishes","Staple Foods","Sweet Snacks/Desserts","Vegetable-Based Dishes"],
            index=None,
            placeholder="Choose an option...",
            )
        
        quantity_ordered = st.number_input("Quantity Ordered")
        payment_date = st.date_input("Payment Date", date.today())

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

        # st.write(col)
        # st.write(new_data[col])
        # st.write(label_encoders_1b[col].classes_)

        st.success(f"Prediction Profit: {prediction_regressor}")
        #st.subheader(f"Predicted Percentage Profit per Unit: {prediction_regressor:.2f}%")

# -----------------------------
# Recommender
# -----------------------------

def normalize_the_rules(loaded_rules):
   
    def convert_to_frozenset(text):
        if isinstance(text, frozenset):
            return frozenset(i.lower().strip() for i in text)
        elif isinstance(text, str):
            # Convert string like "frozenset({'whole milk', 'other vegetables'})"
            import ast
            return frozenset(i.lower().strip() for i in ast.literal_eval(text.replace("frozenset", "")))
        else:
            raise ValueError("Unexpected type in rules")
    
    loaded_rules['antecedents'] = loaded_rules['antecedents'].apply(convert_to_frozenset)
    loaded_rules['consequents'] = loaded_rules['consequents'].apply(convert_to_frozenset)
    
    return loaded_rules

def dynamic_recommender_intermediate(cart, rules_df):
    # Convert cart to set for subset matching
    cart_set = set(i.lower().strip() for i in cart)

    matching_rules = rules_df[
        rules_df['antecedents'].apply(lambda x: set(x).issubset(cart_set))
    ]

    # If no rules match
    if matching_rules.empty:
        return "No recommendation available."
    
    # Sort rules by confidence (highest first)
    matching_rules = matching_rules.sort_values(by='confidence', ascending=False)
    
    
    # Collect recommendations with ranking
    recommendations = []
    seen_items = set()
    
    for _, row in matching_rules.iterrows():
        for item in row['consequents']:
            # Avoid recommending items already in cart
            if item not in cart_set and item not in seen_items:
                recommendations.append(item)
                seen_items.add(item)
    
    # If all consequents were already in cart
    if not recommendations:
        return "No recommendation available."
    
    return recommendations

with tab3:

    st.write("Association rule-based product suggestions.")

    with st.form("recommender_form"):

        

        # Simple multi-select for mockup
        all_items = ["whole milk", "yogurt", "rolls/buns", "soda", "bottled water", "tropical fruit"]
        basket = st.multiselect("Select items in basket", all_items)


        submit_recommendations = st.form_submit_button("Get Recommendations")

    if submit_recommendations:

        #we have to clean the rules first as the frozen set part affects it
        loaded_rules = recommender_rules
        clean_loaded_rules = normalize_the_rules( loaded_rules)

        recommendations = dynamic_recommender_intermediate(basket, clean_loaded_rules)

        st.success(f"Recommendations: {recommendations}")



