from flask import Flask, request, jsonify
# Cross-Origin Resource Sharing (CORS)
# Modern browsers apply the "same-origin policy", which blocks web pages from
# making requests to a different origin than the one that served the page.
# This helps prevent malicious sites from reading sensitive data from another
# site you are logged into.
#
# However, there are many legitimate cases where cross-origin requests are
# needed. One example is:
#
## Single-Page Applications (SPA) hosted at example-frontend.com need to call
## APIs hosted at api.example-backend.com.
#
# To support this safely, CORS lets servers explicitly allow such requests.
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
# CORS(
#     app,
#     resources={r"/api/*": {
#         "origins": [
#             "https://127.0.0.1",
#             "https://localhost"
#         ]
#     }},
#     methods=["GET", "POST", "OPTIONS"],
#     allow_headers=["Content-Type"]
# )

CORS(
    app, supports_credentials=False,
    resources={r"/api/*": { # This means CORS will only apply to routes that start with /api/
               "origins": [
                   "https://127.0.0.1", "https://localhost",
                   "https://127.0.0.1:443", "https://localhost:443",
                   "http://127.0.0.1", "http://localhost",
                   "http://127.0.0.1:5000", "http://localhost:5000",
                   "http://127.0.0.1:5500", "http://localhost:5500"
                ]
    }},
    methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"])

# CORS(app, supports_credentials=False,
#      origins=["*"])

# Load different models
# joblib is used to load a trained model so that the API can serve ML predictions
# baseline models
decisiontree_classifier_baseline = joblib.load('./model/decisiontree_classifier_baseline.pkl')
decisiontree_regressor_optimum = joblib.load('./model/decisiontree_regressor_optimum.pkl')
label_encoders_1b = joblib.load('./model/label_encoders_1b.pkl')

# intermediate/additional classifiers
naive_bayes_classifier_optimum = joblib.load('./model/naive_Bayes_classifier_optimum.pkl')
knn_classifier_optimum = joblib.load('./model/knn_classifier_optimum.pkl')
random_forest_classifier_optimum = joblib.load('./model/random_forest_classifier_optimum.pkl')
support_vector_classifier_optimum = joblib.load('./model/support_vector_classifier_optimum.pkl')

# advanced models/data for optional endpoints
# k-Means cluster model (trained elsewhere and saved to disk)
kmeans_model = joblib.load('./model/kmeans_model.pkl')
# association rules dictionary: antecedent tuple -> list of consequent tuples
assoc_rules = joblib.load('./model/assoc_rules.pkl')


# Helper utilities -----------------------------------------------------------
def _validate_numeric_inputs(data, fields):
    """Ensure that each expected field is present and can be interpreted as a number.
    Returns (valid: bool, message: str|None).
    """
    missing = [f for f in fields if data.get(f) is None]
    if missing:
        return False, f"Missing field(s): {', '.join(missing)}"
    return True, None


def _validate_items_list(data):
    """Check that request JSON contains a list under key 'items'."""
    items = data.get('items')
    if not isinstance(items, list):
        return False, "Missing or invalid 'items'; expected a list of product identifiers"
    return True, None


def _predict_from_model(model, data):
    """Run prediction against a simple numeric-feature model.
    Assumes all features are numeric and provided in 'data'."""
    expected_features = ['monthly_fee', 'customer_age', 'support_calls']
    valid, msg = _validate_numeric_inputs(data, expected_features)
    if not valid:
        return None, msg

    # build dataframe keeping column order consistent with training
    new_data = pd.DataFrame([{f: data.get(f) for f in expected_features}])
    new_data = new_data[expected_features]
    pred = model.predict(new_data)[0]
    return int(pred), None


# Defines an HTTP endpoint for the baseline decision‑tree classifier
@app.route('/api/v1/models/decision-tree-classifier/predictions', methods=['POST'])
def predict_decision_tree_classifier():
    data = request.get_json()
    prediction, error = _predict_from_model(decisiontree_classifier_baseline, data)
    if error:
        return jsonify({'error': error}), 400
    return jsonify({'Predicted Class = ': prediction})

# *1* Sample JSON POST values
# {
#     "monthly_fee": 60,
#     "customer_age": 30,
#     "support_calls": 1
# }

# additional classifier endpoints for intermediate level
@app.route('/api/v1/models/naive-bayes-classifier/predictions', methods=['POST'])
def predict_naive_bayes():
    data = request.get_json()
    prediction, error = _predict_from_model(naive_bayes_classifier_optimum, data)
    if error:
        return jsonify({'error': error}), 400
    return jsonify({'Predicted Class = ': prediction})


@app.route('/api/v1/models/knn-classifier/predictions', methods=['POST'])
def predict_knn():
    data = request.get_json()
    prediction, error = _predict_from_model(knn_classifier_optimum, data)
    if error:
        return jsonify({'error': error}), 400
    return jsonify({'Predicted Class = ': prediction})


@app.route('/api/v1/models/random-forest-classifier/predictions', methods=['POST'])
def predict_random_forest():
    data = request.get_json()
    prediction, error = _predict_from_model(random_forest_classifier_optimum, data)
    if error:
        return jsonify({'error': error}), 400
    return jsonify({'Predicted Class = ': prediction})


@app.route('/api/v1/models/svm-classifier/predictions', methods=['POST'])
def predict_svm():
    data = request.get_json()
    prediction, error = _predict_from_model(support_vector_classifier_optimum, data)
    if error:
        return jsonify({'error': error}), 400
    return jsonify({'Predicted Class = ': prediction})

# --- advanced endpoints ----------------------------------------------------
@app.route('/api/v1/models/kmeans-cluster/predictions', methods=['POST'])
def predict_kmeans_cluster():
    """Return predicted cluster index for numeric data (same features as classifier)."""
    data = request.get_json()
    prediction, error = _predict_from_model(kmeans_model, data)
    if error:
        return jsonify({'error': error}), 400
    return jsonify({'cluster': prediction})


@app.route('/api/v1/recommendations', methods=['POST'])
def recommend_products():
    """Simple recommender using precomputed association rules.

    Expects JSON body with key 'items' containing a list of previously
    purchased products. Returns a list of recommended products.
    """
    data = request.get_json()
    valid, msg = _validate_items_list(data)
    if not valid:
        return jsonify({'error': msg}), 400

    items = data.get('items', [])
    recs = set()
    for antecedent, consequents in assoc_rules.items():
        if set(antecedent).issubset(items):
            for cons in consequents:
                # cons may be tuple
                if isinstance(cons, (list, tuple)):
                    recs.update(cons)
                else:
                    recs.add(cons)
    return jsonify({'recommendations': list(recs)})


# *2.a.* Sample cURL POST values (without HTTPS in NGINX and Gunicorn)

# curl -X POST http://127.0.0.1:5000/api/v1/models/decision-tree-classifier/predictions \
#   -H "Content-Type: application/json" \
#   -d "{\"monthly_fee\": 60, \"customer_age\": 30, \"support_calls\": 1}"

# *2.b.* Sample cURL POST values (with HTTPS in NGINX and Gunicorn)

# curl --insecure -X POST https://127.0.0.1/api/v1/models/decision-tree-classifier/predictions \
#   -H "Content-Type: application/json" \
#   -d "{\"monthly_fee\": 60, \"customer_age\": 30, \"support_calls\": 1}"

# *3* Sample PowerShell values:

# $body = @{
#     monthly_fee = 60
#     customer_age = 30
#     support_calls = 1
# } | ConvertTo-Json

# Invoke-RestMethod -Uri http://127.0.0.1:5000/api/v1/models/decision-tree-classifier/predictions `
#     -Method POST `
#     -Body $body `
#     -ContentType "application/json"

@app.route('/api/v1/models/decision-tree-regressor/predictions', methods=['POST'])
def predict_decision_tree_regressor():
    data = request.get_json()
    # Expected input keys:
    # 'PaymentDate', 'CustomerType', 'BranchSubCounty',
    # 'ProductCategoryName', 'QuantityOrdered', 'PercentageProfitPerUnit'

    # Create a DataFrame based on the input
    new_data = pd.DataFrame([data])

    # Convert PaymentDate to datetime
    new_data['PaymentDate'] = pd.to_datetime(new_data['PaymentDate'])

    # Identify all datetime columns
    datetime_columns = new_data.select_dtypes(include=['datetime64']).columns

    categorical_cols = new_data.select_dtypes(exclude=['int64', 'float64', 'datetime64[ns]']).columns

    # Encode categorical columns
    for col in categorical_cols:
        if col in new_data:
            new_data[col] = label_encoders_1b[col].transform(new_data[col])

    # Feature engineering for date
    new_data['PaymentDate_year'] = new_data['PaymentDate'].dt.year # type: ignore
    new_data['PaymentDate_month'] = new_data['PaymentDate'].dt.month # type: ignore
    new_data['PaymentDate_day'] = new_data['PaymentDate'].dt.day # type: ignore
    new_data['PaymentDate_dayofweek'] = new_data['PaymentDate'].dt.dayofweek # type: ignore
    new_data = new_data.drop(columns=datetime_columns)

    # Define the expected feature order (based on the order used during training)
    expected_features = [
        'CustomerType',
        'BranchSubCounty',
        'ProductCategoryName',
        'QuantityOrdered',
        'PaymentDate_year',
        'PaymentDate_month',
        'PaymentDate_day',
        'PaymentDate_dayofweek'
    ]

    # Reorder and select only the expected columns
    new_data = new_data[expected_features]

    # Predict
    prediction = decisiontree_regressor_optimum.predict(new_data)[0]
    return jsonify({'Predicted Percentage Profit per Unit = ': float(prediction)})

# *1* Sample JSON POST values
# {
#     "CustomerType": "Business",
#     "BranchSubCounty": "Kilimani",
#     "ProductCategoryName": "Meat-Based Dishes",
#     "QuantityOrdered": 8,
#     "PaymentDate": "2027-11-13"
# }

# *2.a.* Sample cURL POST values

# curl -X POST http://127.0.0.1:5000/api/v1/models/decision-tree-regressor/predictions \
#   -H "Content-Type: application/json" \
#   -d "{\"CustomerType\": \"Business\",
# 	\"BranchSubCounty\": \"Kilimani\",
# 	\"ProductCategoryName\": \"Meat-Based Dishes\",
# 	\"QuantityOrdered\": 8,
# 	\"PaymentDate\": \"2027-11-13\"}"

# *2.b.* Sample cURL POST values

# curl --insecure -X POST https://127.0.0.1/api/v1/models/decision-tree-regressor/predictions \
#   -H "Content-Type: application/json" \
#   -d "{\"CustomerType\": \"Business\",
# 	\"BranchSubCounty\": \"Kilimani\",
# 	\"ProductCategoryName\": \"Meat-Based Dishes\",
# 	\"QuantityOrdered\": 8,
# 	\"PaymentDate\": \"2027-11-13\"}"

# *3* Sample PowerShell values:

# $body = @{
#     PaymentDate         = "2027-11-13"
#     CustomerType        = "Business"
#     BranchSubCounty     = "Kilimani"
#     ProductCategoryName = "Meat-Based Dishes"
#     QuantityOrdered = 8
# } | ConvertTo-Json

# Invoke-RestMethod -Uri http://127.0.0.1:5000/api/v1/models/decision-tree-regressor/predictions `
#     -Method POST `
#     -Body $body `
#     -ContentType "application/json"

# This ensures the Flask web server only starts when you run this file directly
# (e.g., `python api.py`), and not if you import api.py from another script or test.

# __name__ is a special variable in Python. When you run a script directly,
# __name__ is set to '__main__'. If the script is imported, __name__ is set to
# the module's name.

# if __name__ == '__main__': checks if the script is being run directly.

# app.run(debug=True) starts the Flask development server with debugging enabled.
# This means:
## The server will automatically reload if you make code changes.
## You get detailed error messages in the browser if something goes wrong.
if __name__ == '__main__':
    app.run(debug=True)
# if __name__ == '__main__':
#     app.run(debug=False)
# if __name__ == "__main__":
#     app.run(ssl_context=("cert.pem", "key.pem"), debug=True)
