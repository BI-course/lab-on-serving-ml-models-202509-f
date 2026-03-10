import streamlit as st
import joblib
import numpy as np
import pandas as pd
from datetime import date
import ast

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Advanced ML Serving Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium look
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { width: 100%; border-radius: 8px; height: 3em; background-color: #4A90E2; color: white; }
    .stMetric { background-color: white; padding: 15px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

# -----------------------------
# Model Loading (Cached)
# -----------------------------
@st.cache_resource
def load_models():
    models = {
        "dt_classifier": joblib.load("./model/decisiontree_classifier_baseline.pkl"),
        "dt_regressor": joblib.load("./model/decisiontree_regressor_optimum.pkl"),
        "label_encoders_1b": joblib.load("./model/label_encoders_1b.pkl"),
        "knn_optimum": joblib.load("./model/knn_classifier_optimum.pkl"),
        "nb_optimum": joblib.load("./model/naive_Bayes_classifier_optimum.pkl"),
        "rf_optimum": joblib.load("./model/random_forest_classifier_optimum.pkl"),
        "svm_optimum": joblib.load("./model/support_vector_classifier_optimum.pkl"),
        "scaler_4": joblib.load("./model/scaler_4.pkl"),
        "scaler_5": joblib.load("./model/scaler_5.pkl"),
        "label_encoders_4": joblib.load("./model/label_encoders_4.pkl"),
        "label_encoders_5": joblib.load("./model/label_encoders_5.pkl"),
        "kmeans": joblib.load("./model/kmeans_model.pkl"),
        "apriori": pd.read_csv("./model/top_rules_7b.csv")
    }
    return models

try:
    models = load_models()
except Exception as e:
    st.error(f"Error loading models: {e}. Please ensure all .pkl files are in the /model directory.")
    st.stop()

# -----------------------------
# Sidebar Navigation
# -----------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Shopper Intent (Classifiers)", "Customer Segmentation (Clustering)", "Product Recommender (Apriori)", "Business Metrics (Regression)"])

# -----------------------------
# Home Page
# -----------------------------
if page == "Home":
    st.title("🚀 ML Serving Dashboard")
    st.markdown("""
    Welcome to the **Unified Machine Learning Serving Dashboard**. 
    This application demonstrates the practical application of various ML models including:
    - **Classification**: Churn prediction and Shopper Intent analysis.
    - **Regression**: Business profit prediction.
    - **Clustering**: Customer segmentation for target marketing.
    - **Association Rules**: Product recommendations.
    
    *Developed by Group Member 1 & 3.*
    """)
    st.image("https://images.unsplash.com/photo-1551288049-bbbda536338a?ixlib=rb-1.2.1&auto=format&fit=crop&w=1350&q=80")

# -----------------------------
# Shopper Intent Page
# -----------------------------
elif page == "Shopper Intent (Classifiers)":
    st.title("🛍️ Shopper Intent Classifiers")
    st.write("Compare different classification models on online shopper data.")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Session Details")
        admin = st.number_input("Administrative Pages", 0, 30, 0)
        info = st.number_input("Informational Pages", 0, 30, 0)
        prod = st.number_input("Product Related Pages", 0, 500, 20)
        page_val = st.number_input("Page Values", 0.0, 500.0, 0.0)
        bounce = st.number_input("Bounce Rates", 0.0, 1.0, 0.0)
        exit_r = st.number_input("Exit Rates", 0.0, 1.0, 0.02)
        month = st.selectbox("Month", ["Feb", "Mar", "May", "Oct", "Nov"])
        visitor = st.selectbox("Visitor Type", ["Returning_Visitor", "New_Visitor"])
        weekend = st.selectbox("Weekend", ["FALSE", "TRUE"])
        
        predict_btn = st.button("Run Comparison")

    with col2:
        st.subheader("Model Agreement")
        if predict_btn:
            # Preprocessing for Shoppers
            input_data = {
                'Administrative': admin, 'Administrative_Duration': 0,
                'Informational': info, 'Informational_Duration': 0,
                'ProductRelated': prod, 'ProductRelated_Duration': 100,
                'BounceRates': bounce, 'ExitRates': exit_r, 
                'PageValues': page_val, 'SpecialDay': 0, 'Month': month,
                'OperatingSystems': 1, 'Browser': 1, 'Region': 1, 
                'TrafficType': 1, 'VisitorType': visitor, 'Weekend': weekend
            }
            df = pd.DataFrame([input_data])
            
            # Helper for encoding/scaling
            def process(df, encoders, scaler):
                temp_df = df.copy()
                for col in ['VisitorType', 'Weekend', 'Month']:
                    temp_df[col] = encoders[col].transform(temp_df[col].astype(str))
                return scaler.transform(temp_df)

            processed_4 = process(df, models["label_encoders_4"], models["scaler_4"])
            processed_5 = process(df, models["label_encoders_5"], models["scaler_5"])
            
            # Predictions
            preds = {
                "Naive Bayes": models["nb_optimum"].predict(processed_4)[0],
                "kNN": models["knn_optimum"].predict(processed_4[:, :8])[0],
                "SVM": models["svm_optimum"].predict(processed_5)[0],
                "Random Forest": models["rf_optimum"].predict(processed_4)[0]
            }
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Naive Bayes", "BUY" if preds["Naive Bayes"] else "NO")
            m2.metric("kNN", "BUY" if preds["kNN"] else "NO")
            m3.metric("SVM", "BUY" if preds["SVM"] else "NO")
            m4.metric("Random Forest", "BUY" if preds["Random Forest"] else "NO")
            
            # Chart
            chart_data = pd.DataFrame({
                "Model": list(preds.keys()),
                "Prediction": list(preds.values())
            })
            st.bar_chart(chart_data.set_index("Model"))

# -----------------------------
# Clustering Page
# -----------------------------
elif page == "Customer Segmentation (Clustering)":
    st.title("🎯 Customer Segmentation")
    st.write("Using k-Means to identify customer clusters based on behavior.")
    
    age = st.slider("Age", 18, 100, 30)
    income = st.slider("Annual Income (k$)", 10, 200, 50)
    score = st.slider("Spending Score (1-100)", 1, 100, 50)
    
    if st.button("Identify Segment"):
        X = pd.DataFrame([{'Age': age, 'Annual Income (k$)': income, 'Spending Score (1-100)': score}])
        cluster_id = int(models["kmeans"].predict(X)[0])
        
        descriptions = {
            0: "Targeted Premium: Young, high income, high spending.",
            1: "Average Spenders: Young, average metrics.",
            2: "Low Spenders: Mature, high income, low spending.",
            3: "Frugal: Mature, low income, low spending.",
            4: "Luxury Shoppers: Middle-aged, medium income, high spending."
        }
        
        st.markdown(f"### Predicted Cluster: **{cluster_id}**")
        st.info(descriptions.get(cluster_id, "Unknown Segment"))

# -----------------------------
# Recommender Page
# -----------------------------
elif page == "Product Recommender (Apriori)":
    st.title("🛒 Smart Recommender")
    st.write("Association rule-based product suggestions.")
    
    # Simple multi-select for mockup
    all_items = ["whole milk", "yogurt", "rolls/buns", "soda", "bottled water", "tropical fruit"]
    basket = st.multiselect("Select items in basket", all_items)
    
    if st.button("Get Recommendations"):
        if not basket:
            st.warning("Basket is empty!")
        else:
            recs = []
            basket_set = set(basket)
            
            for _, row in models["apriori"].iterrows():
                def parse_set(s):
                    inner = s.replace("frozenset({", "").replace("})", "")
                    try:
                        return set(ast.literal_eval(f"{{{inner}}}"))
                    except:
                        return set([i.strip().strip("'").strip('"') for i in inner.split(',')])
                
                ants = parse_set(row['antecedents'])
                cons = parse_set(row['consequents'])
                
                if ants.issubset(basket_set):
                    recs.extend(list(cons))
            
            unique_recs = [r for r in set(recs) if r not in basket_set]
            if unique_recs:
                st.success("We recommend:")
                for r in unique_recs[:5]:
                    st.write(f"- {r}")
            else:
                st.write("No strong recommendations found.")

# -----------------------------
# Regression Page
# -----------------------------
elif page == "Business Metrics (Regression)":
    st.title("📈 Profit Prediction")
    st.write("Decision Tree Regressor for business profit estimation.")
    
    with st.form("profit_form"):
        c_type = st.selectbox("Customer Type", ["Business", "Individual"])
        subcounty = st.text_input("Sub County", "Kilimani")
        prod_cat = st.text_input("Product Category", "Meat-Based Dishes")
        qty = st.number_input("Quantity Ordered", 1, 100, 1)
        p_date = st.date_input("Payment Date", date.today())
        
        if st.form_submit_button("Predict Profit"):
            try:
                new_data = pd.DataFrame([{
                    'CustomerType': c_type, 'BranchSubCounty': subcounty,
                    'ProductCategoryName': prod_cat, 'QuantityOrdered': qty,
                    'PaymentDate_year': p_date.year, 'PaymentDate_month': p_date.month,
                    'PaymentDate_day': p_date.day, 'PaymentDate_dayofweek': p_date.weekday()
                }])
                
                # Apply encoding
                for col in ['CustomerType', 'BranchSubCounty', 'ProductCategoryName']:
                    new_data[col] = models["label_encoders_1b"][col].transform(new_data[col])
                
                pred = models["dt_regressor"].predict(new_data)[0]
                st.success(f"### Predicted Percentage Profit: {pred:.2f}%")
            except Exception as e:
                st.error(f"Error: {e}")

st.sidebar.divider()
st.sidebar.info("This is a Streamlit demo for the BI Course project.")
