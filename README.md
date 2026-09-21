# 🤖 Responsible AI Pipeline

> 🚀 An end-to-end **Responsible AI, Machine Learning & Generative AI framework** for evaluating tabular datasets, detecting bias, improving fairness, training ML models, generating explainability insights, and producing automated governance reports.

---

## 🌟 Overview

The **Responsible AI Pipeline** provides an automated workflow for analyzing machine-learning systems from multiple Responsible AI perspectives, including:

* 📊 Data quality
* ⚖️ Fairness and bias
* 🤖 Model performance
* 🔎 Explainability
* ⚠️ Ethical risk
* 🏛️ AI governance

The pipeline combines traditional machine learning, synthetic data generation, fairness analysis, explainable AI, and an LLM-powered decision engine into a single workflow.

### 🔄 High-Level Architecture

```text
📂 Input Dataset
      │
      ▼
🔍 Validation & Analysis
      │
      ▼
🧹 Preprocessing
      │
      ▼
⚖️ Imbalance & Bias Analysis
      │
      ├──────────────► 🔄 SMOTE / SMOTENC
      │
      ├──────────────► 🧬 CTGAN
      │
      ▼
🛠️ Fairness Mitigation
      │
      ▼
🤖 Model Training & Evaluation
      │
      ▼
📈 Responsible AI Metrics
      │
      ▼
🧠 LLM Decision Engine
      │
      ▼
💡 Recommendations & Optimization
      │
      ▼
📊 Visualizations + 📄 PDF Report
```

---

# ✨ Features

| Area                   | Capabilities                                                      |
| ---------------------- | ----------------------------------------------------------------- |
| 📂 **Dataset**         | CSV loading, validation, summary, target detection                |
| 🧹 **Preprocessing**   | Missing values, duplicates, encoding, outliers, skewness          |
| ⚖️ **Imbalance**       | Class distribution analysis, SMOTE, SMOTENC                       |
| 🧬 **Synthetic Data**  | CTGAN-based tabular data generation                               |
| 🔍 **Bias & Fairness** | Bias detection, fairness metrics, proxy-bias analysis, mitigation |
| 🤖 **ML Models**       | Random Forest, Gradient Boosting, Voting models                   |
| 🔎 **Explainability**  | SHAP, LIME, feature importance, LLM explanations                  |
| 🧠 **Generative AI**   | Groq-powered analysis and recommendation engine                   |
| 📈 **RAI Metrics**     | EDQS, ERI, RAI, Fairness, Explainability & Governance             |
| 📊 **Visualization**   | Bias, fairness, metric and model-performance visualizations       |
| 📄 **Reporting**       | Automated Responsible AI PDF reports                              |
| 🖥️ **Dashboard**      | Streamlit-based results visualization                             |

---

# 🛠️ Technology Stack

### 🐍 Core

* Python
* Pandas
* NumPy

### 🤖 Machine Learning

* Scikit-learn
* Random Forest
* Gradient Boosting
* Imbalanced-learn

### ⚖️ Responsible AI

* Fairlearn
* AIF360

### 🧬 Synthetic Data

* CTGAN
* SDV
* Faker

### 🔎 Explainable AI

* SHAP
* LIME

### 🧠 Generative AI

* LangChain
* Groq API

### 📊 Visualization

* Matplotlib
* Seaborn
* Plotly

### 📄 Reporting & Interface

* ReportLab
* FPDF
* Tkinter
* Streamlit

---

# 📁 Project Structure

```text
Responsible-AI-Pipeline/
│
├── 📄 main.py
├── ⚙️ config.py
├── 📦 requirements.txt
├── 📖 README.md
├── 🚫 .gitignore
├── 🔐 .env                    # Local only
│
├── 📦 modules/
│   │
│   ├── ⚖️ balancing/
│   │   ├── imbalance_detector.py
│   │   ├── imbalance_report.py
│   │   ├── smote_module.py
│   │   └── smote_visualizer.py
│   │
│   ├── 🔍 bias/
│   │   ├── bias_detector.py
│   │   ├── bias_table.py
│   │   ├── fairness_fix.py
│   │   ├── fairness_metrics.py
│   │   └── proxy_bias_remover.py
│   │
│   ├── 🖥️ dashboard/
│   │   └── app.py
│   │
│   ├── 📂 dataset/
│   │   ├── loader.py
│   │   ├── summary.py
│   │   └── validator.py
│   │
│   ├── 🔎 explainability/
│   │   └── llm_explainer.py
│   │
│   ├── 🧠 llm/
│   │   ├── apply_recommendations.py
│   │   └── llm_decision_engine.py
│   │
│   ├── 📊 metrics/
│   │   ├── edqs.py
│   │   ├── eri.py
│   │   ├── explainability_score.py
│   │   ├── governance_score.py
│   │   └── rai.py
│   │
│   ├── 🤖 model/
│   │   ├── evaluate_model.py
│   │   ├── save_model.py
│   │   └── train_model.py
│   │
│   ├── 🧹 preprocessing/
│   │   ├── cleaning.py
│   │   ├── encoding.py
│   │   ├── missing_handler.py
│   │   ├── outlier_handler.py
│   │   ├── preprocessing_pipeline.py
│   │   └── skewness_fixer.py
│   │
│   ├── 📄 reporting/
│   │   └── pdf_report.py
│   │
│   ├── 📑 reports/
│   │   └── report_generator.py
│   │
│   ├── 🧬 synthetic/
│   │   ├── ctgan_generator.py
│   │   └── synthetic_generator.py
│   │
│   ├── 🎯 target/
│   │   └── detect_target.py
│   │
│   ├── 🔧 utils/
│   │   ├── helpers.py
│   │   ├── logger.py
│   │   ├── save_metrics.py
│   │   └── system_info.py
│   │
│   └── 📊 visualization/
│       ├── before_after_graphs.py
│       ├── bias_radar_chart.py
│       ├── bias_visualizations.py
│       ├── boxplots.py
│       ├── ctgan_graph.py
│       ├── edqs_graph.py
│       ├── eri_graph.py
│       ├── feature_importance.py
│       ├── heatmaps.py
│       ├── piecharts.py
│       └── rai_graph.py
│
└── 📁 outputs/               # Generated locally
```

---

# ⚙️ Installation

## 1️⃣ Clone the Repository

Replace the URL with your GitHub repository:

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
```

## 2️⃣ Create a Virtual Environment

### 🪟 Windows

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then:

```powershell
.\venv\Scripts\Activate.ps1
```

### 🐧 Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

## 3️⃣ Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

# 🔐 Environment Configuration

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

The application loads environment variables using `python-dotenv`.

### 🚨 Security

Never commit secrets to GitHub.

Recommended `.gitignore`:

```gitignore
.env
venv/
.venv/
__pycache__/
*.pyc
outputs/
*.pkl
```

Check before committing:

```bash
git status
```

---

# ▶️ Running the Pipeline

Start the main pipeline:

```bash
python main.py
```

The application will open the dataset-selection interface.

Select the required `.csv` file and the pipeline will process it automatically.

If the GUI file picker is unavailable, the application can request the dataset path through the terminal.

---

# 📂 Input Dataset

The primary pipeline accepts **CSV tabular datasets**.

The system automatically analyzes:

* 🎯 Target column
* 📊 Task type
* 🔤 Categorical features
* 🔢 Numerical features
* ⚖️ Class distribution
* ❌ Missing values
* 📑 Duplicate records

### Supported Tasks

```text
✅ Classification
✅ Regression
```

---

# 🔄 Processing Pipeline

## 1. 🔍 Dataset Analysis

The dataset is loaded and inspected for:

* Shape and structure
* Data types
* Missing values
* Duplicate records
* Target column
* Feature types
* Class distribution

---

## 2. 🧹 Preprocessing

Depending on the dataset, preprocessing may include:

* Missing-value handling
* Duplicate removal
* Categorical encoding
* Outlier processing
* Skewness correction
* Feature transformation

---

## 3. ⚖️ Imbalance Handling

For classification datasets, the system evaluates class distribution and can apply:

* **SMOTE**
* **SMOTENC**

This helps address imbalanced training data.

---

## 4. 🧬 Synthetic Data Generation

CTGAN can generate synthetic tabular records based on the input dataset.

The pipeline provides comparisons between real and synthetic data through generated visualizations.

---

## 5. 🔍 Bias & Fairness Analysis

The fairness layer evaluates potential disparities across relevant attributes.

It includes:

* Bias detection
* Fairness metrics
* Bias tables
* Proxy-bias detection
* Fairness comparisons
* Visualization of bias-related results

---

## 6. 🛠️ Fairness Mitigation

Where applicable, fairness-related transformations are applied and the resulting metrics are compared with the original results.

```text
Original Data
     ↓
Bias / Fairness Assessment
     ↓
Mitigation
     ↓
Re-evaluation
```

---

## 7. 🤖 Model Training

The project supports several ML approaches.

### 🌲 Random Forest

```text
RandomForestClassifier
RandomForestRegressor
```

### 📈 Gradient Boosting

```text
GradientBoostingClassifier
GradientBoostingRegressor
```

### 🗳️ Voting Models

Multiple trained estimators can be combined using voting-based approaches.

---

# 📊 Model Evaluation

### Classification

The pipeline evaluates:

* Accuracy
* Precision
* Recall
* F1 Score

### Regression

The pipeline evaluates:

* MAE
* MSE
* RMSE
* R²

Model feature importance is also generated where applicable.

---

# 📈 Responsible AI Metrics

The project provides a set of custom metrics to summarize different aspects of the pipeline.

| Metric                      | Purpose                                |
| --------------------------- | -------------------------------------- |
| 📊 **EDQS**                 | Evaluates ethical data quality         |
| ⚠️ **ERI**                  | Summarizes ethical/model risk          |
| 🤖 **RAI**                  | Aggregates Responsible AI dimensions   |
| ⚖️ **Fairness Score**       | Evaluates fairness-related performance |
| 🔎 **Explainability Score** | Evaluates model interpretability       |
| 🏛️ **Governance Score**    | Summarizes governance-related factors  |

### 📊 EDQS — Ethical Data Quality Score

EDQS considers data-quality factors such as:

* Missing values
* Duplicate records
* Class imbalance

Example configuration:

```python
EDQS_WEIGHTS = {
    "missing": 0.4,
    "duplicate": 0.3,
    "imbalance": 0.3
}
```

### ⚠️ ERI — Ethical Risk Index

ERI combines relevant risk factors including fairness, model performance, imbalance and explainability-related information.

### 🤖 RAI — Responsible AI Index

RAI provides an aggregated view of Responsible AI characteristics across the processed workflow.

### 🔎 Explainability Score

Explainability is supported through:

* SHAP
* LIME
* Feature importance
* LLM-generated explanations

### 🏛️ Governance Score

The Governance Score summarizes selected Responsible AI dimensions into an overall governance-oriented assessment.

---

# 🧠 LLM Decision Engine

The project integrates a **Groq-powered LLM Decision Engine** to interpret pipeline results.

The engine can analyze:

### 🔍 Analysis

* Dataset quality
* Bias and fairness
* Ethical risk
* Model performance
* Explainability
* Responsible AI metrics

### 💡 Recommendations

Based on the analysis, the LLM can generate recommendations related to:

* Data quality
* Fairness
* Model performance
* Explainability
* Responsible AI improvements

### 🔄 Optimization

Selected recommendations can be incorporated into the iterative optimization workflow.

---

# ⚡ Optimization

The pipeline supports iterative improvement using configurable parameters:

```python
FAST_MODE = True
TEST_SIZE = 0.20
RANDOM_STATE = 42
MAX_ITERATIONS = 5
PATIENCE = 2
MIN_IMPROVEMENT = 0.001
```

| Parameter         | Purpose                       |
| ----------------- | ----------------------------- |
| `FAST_MODE`       | Enables faster execution      |
| `TEST_SIZE`       | Test-set proportion           |
| `RANDOM_STATE`    | Reproducible results          |
| `MAX_ITERATIONS`  | Maximum optimization cycles   |
| `PATIENCE`        | Early stopping                |
| `MIN_IMPROVEMENT` | Minimum improvement threshold |

---

# 📊 Visualizations

The pipeline can generate visualizations for:

* 📈 Before/after comparisons
* ⚖️ Bias and fairness
* 🔥 Correlation analysis
* 🕸️ Bias radar charts
* 📊 EDQS
* ⚠️ ERI
* 🤖 RAI
* 🌲 Feature importance
* 🧬 Real vs synthetic CTGAN data
* 📦 Distribution and box plots
* 🥧 Class/category distributions

---

# 📄 Generated Reports & Outputs

Generated artifacts are stored inside:

```text
outputs/
```

Typical outputs include:

```text
outputs/
│
├── 📄 all_metrics_log.txt
├── 📄 final_responsible_ai_dataset.csv
├── 📄 llm_analysis.txt
├── 📄 llm_recommendations.txt
├── 📄 metrics.txt
├── 📄 responsible_ai_report.pdf
├── 📄 smote_report.txt
│
├── 📊 graphs/
│   └── <dataset-name>/
│       ├── before_after_*.png
│       ├── bias_before_after.png
│       ├── bias_heatmap.png
│       ├── bias_radar_chart.png
│       ├── correlation_heatmap.png
│       ├── ctgan_real_vs_synthetic.png
│       ├── edqs_comparison.png
│       ├── eri_comparison.png
│       ├── fairness_comparison.png
│       ├── feature_importance.png
│       └── rai_comparison.png
│
└── 🤖 models/
    └── trained_model.pkl
```

> ℹ️ The exact output files depend on the selected dataset and processing path.

---

# 📄 Responsible AI Report

The generated PDF report brings the main results together in one document.

It may contain:

* 📂 Dataset summary
* 📊 Data-quality results
* ⚖️ Fairness results
* ⚠️ Ethical risk
* 🤖 Responsible AI score
* 🔎 Explainability results
* 🏛️ Governance assessment
* 🎯 Model performance
* 🧠 LLM analysis
* 💡 Recommendations
* 📈 Before/after comparisons

---

# 🖥️ Streamlit Dashboard

The project also contains a Streamlit dashboard:

```text
modules/dashboard/app.py
```

Run it with:

```bash
streamlit run modules/dashboard/app.py
```

If the dashboard uses generated pipeline results, run:

```bash
python main.py
```

first.

---

# ⚙️ Configuration

Project-level settings are maintained in:

```text
config.py
```

Example configuration:

```python
TEST_SIZE = 0.2
RANDOM_STATE = 42
SMOTE_THRESHOLD = 0.4
CORRELATION_THRESHOLD = 0.95
```

---

# 🧪 Example Execution

```text
📂 Select Dataset
      ↓
🔍 Analyze & Validate
      ↓
🧹 Preprocess
      ↓
⚖️ Check Imbalance
      ↓
🔄 Balance Dataset
      ↓
🧬 Generate Synthetic Data
      ↓
🔍 Analyze Bias & Fairness
      ↓
🛠️ Apply Mitigation
      ↓
🤖 Train Models
      ↓
📊 Evaluate Performance
      ↓
📈 Calculate RAI Metrics
      ↓
🧠 Generate LLM Analysis
      ↓
💡 Apply Recommendations
      ↓
📊 Create Visualizations
      ↓
📄 Generate Report
```

---

# 🔐 Security & Best Practices

Before pushing the project to GitHub:

### ❌ Do not commit

```text
.env
API keys
Passwords
Database credentials
Private certificates
Session tokens
Local virtual environments
Generated model files containing sensitive data
```

### ✅ Verify your repository

```bash
git status
git ls-files
```

If a secret is accidentally exposed:

1. 🔴 Revoke/rotate the credential immediately.
2. 🧹 Remove it from the repository.
3. 🔄 Update the application with a new credential.
4. 🛡️ Review Git history if the secret was previously committed.

---

# 📌 Responsible AI Disclaimer

This project is intended as an **engineering and research framework** for evaluating Responsible AI characteristics in machine-learning workflows.

The generated metrics and LLM recommendations are **decision-support outputs** and should not be treated as definitive assessments.

Results can vary depending on:

* Dataset composition
* Target definition
* Feature selection
* Protected attributes
* Sampling strategy
* Model architecture
* Evaluation metrics
* Preprocessing decisions
* Fairness definitions

Human review and domain-specific evaluation remain important when interpreting the results.

---

# 🚀 Future Enhancements

* 🌐 Web-based dataset upload
* 📊 Interactive Responsible AI dashboard
* 🔐 Authentication and role-based access
* 🧠 Additional LLM providers
* ⚖️ More fairness mitigation algorithms
* 🤖 Additional ML models
* 🔎 Interactive SHAP visualizations
* 📋 Automated Model Cards
* 🏛️ AI governance audit trails
* 📦 Docker support
* ☁️ Cloud deployment
* 🔄 Experiment tracking
* 📡 REST API
* 📊 Dataset versioning
* 📝 Automated compliance reporting

---

### 💡 Areas of Interest

* 🤖 Artificial Intelligence
* 🧠 Generative AI
* 🔎 Responsible AI
* 📊 Machine Learning
* 💻 Full Stack Development
* 🔗 RAG Systems

---

# ⭐ Contributing

Contributions, suggestions, and improvements are welcome.

```text
🍴 Fork → 🛠️ Modify → 📤 Pull Request
```

For bugs or feature requests, open a GitHub issue.

---

# 📜 License

This project is currently intended for **educational, research, and portfolio purposes**.

If you plan to distribute or use the project commercially, add an appropriate open-source license such as the **MIT License**.

---

# 🚀 Quick Start

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git

# 2. Enter project
cd YOUR_REPOSITORY

# 3. Create virtual environment
python -m venv venv

# 4. Activate - Windows
.\venv\Scripts\Activate.ps1

# 5. Install dependencies
pip install -r requirements.txt

# 6. Create .env
# GROQ_API_KEY=your_groq_api_key_here

# 7. Run pipeline
python main.py

# 8. Optional dashboard
streamlit run modules/dashboard/app.py
```

---

### 🤖 Built with Python • Machine Learning • Generative AI • Responsible AI • Explainable AI
