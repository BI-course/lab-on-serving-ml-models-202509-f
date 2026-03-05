[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/yKzEOXws)
[![Open in Codespaces](https://classroom.github.com/assets/launch-codespace-2972f46106e565e64193e422d61a12cf1da4916b45550586e14ef0a7c637dd04.svg)](https://classroom.github.com/open-in-codespaces?assignment_repo_id=22655376)
# Serving Machine Learning Models through RESTish APIs using Flask in Python

| Key              | Value                                                                                                                                                                                                                                                                                     |
|:-----------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Course Code** | BBT 4206                                                                                                                                                                                                                                                                                  |
| **Course Name** | BBT 4206: Business Intelligence II (Week 4-6 of 13)                                                                                                                                                                                                                                       |
| **Semester**     | January to April 2026                                                                                                                                                                                                                                                                   |
| **Lecturer**     | Allan Omondi                                                                                                                                                                                                                                                                              |
| **Contact**      | aomondi@strathmore.edu                                                                                                                                                                                                                                                                    |
| **Note**         | The lecture contains both theory and practice.<br/>This notebook forms part of the practice.<br/>It is intended for educational purposes only.<br/>Recommended citation: [BibTex](https://raw.githubusercontent.com/course-files/ServingMLModels/refs/heads/main/RecommendedCitation.bib) |

## Technology Stack

<p align="left">
<img src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/python/python-original.svg" width="40"/>
<img src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/git/git-original.svg" width="40"/>
<img src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/linux/linux-original.svg" width="40" />
<img src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/nginx/nginx-original.svg" width="40"/>
<img src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/flask/flask-original.svg" width="40"/>
<img src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/postman/postman-original.svg" width="40"/>
<img src="assets/images/gunicorn-logo-png-transparent.png" width="60"/>
<img src="assets/images/Hf-logo-with-title.svg" width="120"/>
<img src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/streamlit/streamlit-original.svg" width="40" />
<img src="assets/images/Render-logo-Black.png" width="100"/>
</p>

## Repository Structure

```text
.
├── LICENSE
├── Procfile
├── README.md
├── RecommendedCitation.bib
├── admin_instructions
│   ├── instructions_for_postlab_cleanup.md
│   ├── instructions_for_project_setup.md
│   └── instructions_for_python_installation.md
├── api.py
├── app_server_reverse_proxy_server_setup.md
├── assets
│   └── images
│       ├── Hf-logo-with-title.svg
│       ├── Render-logo-Black.png
│       ├── Streamlit-logo-primary-colormark-darktext.png
│       └── ssh_student_at_localhost_p_2222.jpeg
├── cleanup_instructions.md
├── container-volumes
│   ├── nginx
│   │   └── nginx.conf
│   └── ubuntu
├── docker-compose-dev.yaml
├── docker-compose-prod.yaml
├── docker-compose.yaml
├── dockerfiles
│   ├── Dockerfile.flask-gunicorn-app
│   ├── Dockerfile.nginx
│   └── ubuntu
│       ├── Dockerfile.ubuntu
│       └── entrypoint.sh
├── env.example
├── frontend
│   ├── Proxies.png
│   ├── RequestFlow.jpg
│   ├── RequestFlow.png
│   ├── api_consumer.py
│   ├── api_consumer_from_dev_flask.py
│   ├── api_test_DT_classifier.html
│   ├── api_test_DT_regressor.html
│   └── index.html
├── huggingface-spaces-using-gradio
│   ├── app.py
│   └── requirements.txt
├── lab_submission_instructions.md
├── model
│   ├── decisiontree_classifier_baseline.pkl
│   ├── decisiontree_regressor_optimum.pkl
│   ├── knn_classifier_optimum.pkl
│   ├── label_encoders_1b.pkl
│   ├── label_encoders_2.pkl
│   ├── label_encoders_4.pkl
│   ├── label_encoders_5.pkl
│   ├── naive_Bayes_classifier_optimum.pkl
│   ├── onehot_encoder_3.pkl
│   ├── random_forest_classifier_optimum.pkl
│   ├── scaler_4.pkl
│   ├── scaler_5.pkl
│   └── support_vector_classifier_optimum.pkl
├── publicly_serving_the_model_for_validation_by_domain_experts.md
├── requirements
│   ├── base.txt
│   ├── colab.txt
│   ├── constraints.txt
│   ├── dev.inferred.txt
│   ├── dev.lock.txt
│   ├── dev.txt
│   └── prod.txt
├── rules
├── runtime.txt
└── streamlit-sharing-using-streamlit
    ├── app.py
    └── requirements.txt

15 directories, 58 files
```

## Setup Instructions

- [Setup Instructions](./admin_instructions/instructions_for_project_setup.md)

## Lab Manual

Refer to the files below, in the order specified, for more details:

1. [api_consumwhen er.py](frontend/api_consumer.py)
2. [api.py](api.py)
3. [api_consumer_from_dev_flask.py](frontend/api_consumer_from_dev_flask.py)
4. [api_test_DT_classifier.html](frontend/api_test_DT_classifier.html)

## API Usage Examples

### Classification models

Below are sample cURL commands to call some of the endpoints. The JSON body should contain three numeric fields:
`monthly_fee`, `customer_age`, and `support_calls`.

```bash
curl -X POST http://127.0.0.1:5000/api/v1/models/naive-bayes-classifier/predictions \
  -H "Content-Type: application/json" \
  -d '{"monthly_fee": 60, "customer_age": 30, "support_calls": 1}'
```

Replace `naive-bayes-classifier` with `knn-classifier`, `random-forest-classifier`,
`svm-classifier`, or `decision-tree-classifier` as desired.

### Cluster prediction

The k‑Means cluster model uses the same three numeric inputs.

```bash
curl -X POST http://127.0.0.1:5000/api/v1/models/kmeans-cluster/predictions \
  -H "Content-Type: application/json" \
  -d '{"monthly_fee": 60, "customer_age": 30, "support_calls": 1}'
```

### Recommendations

Send a list of previously purchased `items` (strings). The API returns
products that commonly appear together with the supplied basket.

```bash
curl -X POST http://127.0.0.1:5000/api/v1/recommendations \
  -H "Content-Type: application/json" \
  -d '{"items": ["milk", "bread"]}'
```


*Note:* the models and rules are loaded from the `model/` directory. Make sure
`kmeans_model.pkl` and `assoc_rules.pkl` exist before calling the corresponding
endpoints.
5. [api_test_DT_regressor.html](frontend/api_test_DT_regressor.html)
6. [Reverse Proxy Server and Application Server Setup](app_server_reverse_proxy_server_setup.md)
7. [Publicly Serving the Model for Validation by Domain Experts](publicly_serving_the_model_for_validation_by_domain_experts.md)

## API Endpoints (examples)

The Flask application exposes several `/api/v1/models/.../predictions` endpoints.
Each endpoint accepts a POST request with JSON body containing:

```json
{
  "monthly_fee": <number>,
  "customer_age": <number>,
  "support_calls": <number>
}
```

Replace the model name in the URL to invoke a specific classifier:

| Model                                 | URL suffix                             |
|---------------------------------------|----------------------------------------|
| Decision‑tree classifier (baseline)   | `decision-tree-classifier`             |
| Naive Bayes classifier                | `naive-bayes-classifier`               |
| k‑Nearest neighbours classifier       | `knn-classifier`                       |
| Random forest classifier              | `random-forest-classifier`             |
| Support‑vector‑machine classifier     | `svm-classifier`                       |

Example `curl` command (without HTTPS):

```bash
curl -X POST http://127.0.0.1:5000/api/v1/models/knn-classifier/predictions \
  -H "Content-Type: application/json" \
  -d '{"monthly_fee":60, "customer_age":30, "support_calls":1}'
```

The response is JSON containing the predicted class.


## Lab Submission Instructions

- [Lab Submission Instructions](lab_submission_instructions.md)

## Cleanup Instructions (to be done after submitting the lab)

- [Cleanup Instructions](/admin_instructions/instructions_for_postlab_cleanup.md)
