# PipeGuard-AI


### Water Distribution Pipeline Failure Prediction & Maintenance Prioritization

PipeGuard AI is a data science project that investigates whether machine learning can identify water distribution pipelines at higher risk of failure within the next 12 months.

The project follows the CRISP-DM methodology and includes a publicly deployed Streamlit application.

## 🌐 Live Application

https://pipeguard-ai.streamlit.app

## 1. Business Problem

Water utilities must manage aging infrastructure with limited inspection and maintenance resources.

PipeGuard AI explores how machine learning risk rankings could help maintenance planners prioritize pipeline inspections.

**Research question:**

Can pipeline characteristics and historical conditions be used to predict recorded or simulated failures within the following 12 months?

## 2. Dataset

The project uses a hybrid/synthetic dataset informed by water mains inventory and historical water main breaks data from Ontario, Canada.

- Original records: 200,448
- Original attributes: 27
- Exact duplicates removed: 305
- Records after duplicate removal: 200,143
- Target variable: `failure_next_12_months`
- Prediction task: Binary classification

The dataset contains pipeline characteristics, environmental conditions, operating measurements, and maintenance history.

The target outcomes and augmented records include simulated data. Therefore, the findings are experimental and should not be interpreted as validated real-world failure predictions.

## 3. CRISP-DM Methodology

### Business Understanding
Define the infrastructure maintenance problem, stakeholders, prediction objective, and inspection-prioritization goal.

### Data Understanding
Explore the dataset, attributes, target distribution, missing values, duplicates, and descriptive statistics.

### Data Preparation
Remove duplicates, handle invalid measurements, standardize categories, and construct preprocessing pipelines for numerical and categorical features.

### Modeling
Train and compare three classification algorithms:

1. Logistic Regression
2. Random Forest
3. Gradient Boosting

### Evaluation
Use chronological splitting, walk-forward cross-validation, hyperparameter tuning, and a held-out test period.

Evaluate models with Average Precision, precision, recall, F1-score, and failure capture among the highest-ranked 5% of pipelines.

### Deployment
Develop and publish an interactive Streamlit application that supports experimental risk assessment and inspection planning.

## 4. Model Development

The dataset was divided chronologically:

| Dataset | Assessment years | Records |
|---|---|---:|
| Training | 2009–2023 | 171,879 |
| Validation | 2024 | 14,055 |
| Testing | 2025 | 14,209 |

The selected model is Gradient Boosting with 22 input features.

Selected hyperparameters:

- `learning_rate`: 0.05
- `max_depth`: 3
- `n_estimators`: 100

Preprocessing uses scikit-learn pipelines for imputation, scaling, and categorical encoding.

## 5. Final Model Results

### 2025 Holdout Evaluation

| Metric | Result |
|---|---:|
| Average Precision | 0.0209 |
| Baseline Average Precision | 0.0065 |
| Simulated failures | 93 |
| Pipelines selected (top 5%) | 711 |
| Failures captured | 17 |
| Failure capture rate | 18.28% |
| Top-5% precision | 2.39% |
| Lift over random selection | 3.65× |

The model ranks pipelines above random-selection performance, but its effectiveness is limited.

In an additional age-only benchmark, ranking pipelines by age captured the same number of 2025 failures within the top 5%. The model therefore did not demonstrate an improvement over that simple benchmark at the tested inspection capacity.

These results come from hybrid/synthetic data and are not operational performance guarantees.

## 6. Streamlit Website

The application contains five pages:

### 📊 Risk Dashboard
Displays pipeline statistics, material distributions, experimental risk scores, and top inspection priorities.

### 🔎 Pipeline Explorer
Allows users to search pipeline records, apply filters, and inspect individual asset characteristics.

### 📋 Inspection Prioritization
Allows maintenance planners to adjust inspection capacity, view ranked pipelines, and download inspection lists as CSV files.

### 🧠 Risk Prediction
Uses the saved Gradient Boosting pipeline to calculate an experimental risk score from adjustable pipeline characteristics.

### 📈 Model Information
Displays the modeling approach, validation results, cross-validation, hyperparameter tuning, and limitations.

The website uses a custom navy-and-teal Streamlit theme.

## 7. Technologies

- Python
- pandas and NumPy
- scikit-learn
- Altair
- Streamlit
- Jupyter Notebook
- VS Code
- Git and GitHub

## 8. Project Structure

```text
PipeGuard-AI/
├── .streamlit/
│   └── config.toml
├── DATA/
│   ├── PipeGuard_Main_Dataset.csv.gz
│   └── pipeguard_app_data.csv.gz
├── models/
│   ├── pipeguard_gradient_boosting.joblib
│   └── feature_columns.joblib
├── PipeGuard_AI_Project.ipynb
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## 9. How to Run Locally

Clone the repository:

```bash
git clone https://github.com/ShaimaaAlali8/PipeGuard-AI.git
cd PipeGuard-AI
```

Create and activate a virtual environment:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Start the application:

```bash
python -m streamlit run app.py
```

Open the local address shown in the terminal, usually:

http://localhost:8501

Private-repository access is required to clone the project.

## 10. Limitations & Ethical Considerations

- The dataset contains synthetic/hybrid records and simulated failure outcomes.
- Model scores are not calibrated real-world failure probabilities.
- A high score does not guarantee that a pipeline will fail.
- The selected model does not outperform the age-only benchmark on the tested top-5% capture measure.
- Deployment on verified operational utility data would require additional validation, data governance, and calibration.
- Infrastructure maintenance decisions should involve qualified professionals rather than relying solely on model scores.

## 11. Team Contributions

Add team members and their verified contributions here.

## 12. Data Sources and AI Use

The project was informed by water mains inventory and water main breaks datasets. Add the original dataset URLs, publishers, and applicable usage or licensing information after verification.

AI-assisted coding tools, including ChatGPT and Codex, were used during programming, debugging, and application development. The team is responsible for reviewing and understanding the implementation. Edit this disclosure to reflect the team's actual use and course requirements.

## 13. Project Status

The Streamlit research prototype has been deployed and tested. Its results are intended for academic demonstration only.
