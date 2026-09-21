# 🤖 Responsible AI Pipeline

> 🚀 An end-to-end **Responsible AI, Machine Learning & Generative AI pipeline** for evaluating data quality, detecting bias, improving fairness, training ML models, generating explainability insights, calculating Responsible AI metrics, and producing automated reports.

---

## 🌟 Overview

The **Responsible AI Pipeline** is a Python-based framework designed to evaluate and improve the reliability, fairness, transparency, and governance of machine-learning workflows.

The pipeline takes a tabular dataset and processes it through multiple stages including:

* 📂 Dataset loading and validation
* 🧹 Data cleaning and preprocessing
* ⚖️ Class imbalance detection
* 🔄 SMOTE-based balancing
* 🧬 CTGAN synthetic data generation
* 🔍 Bias and fairness analysis
* 🛠️ Fairness improvement
* 🤖 Machine learning model training
* 📊 Model evaluation
* 🔎 Explainability analysis
* 📈 Responsible AI metrics
* 🧠 LLM-powered analysis and recommendations
* 📄 Automated PDF report generation

### 🔄 Complete Pipeline

```text
📂 CSV Dataset
      ↓
🔍 Dataset Loading & Validation
      ↓
🧹 Data Cleaning & Preprocessing
      ↓
⚖️ Imbalance Detection
      ↓
🔄 SMOTE Balancing
      ↓
🧬 CTGAN Synthetic Data Generation
      ↓
🔍 Bias & Fairness Analysis
      ↓
🛠️ Fairness / Bias Mitigation
      ↓
🤖 Model Training
      ↓
📊 Model Evaluation
      ↓
📈 Responsible AI Metrics
      ↓
🧠 LLM Decision Engine
      ↓
💡 AI Recommendations
      ↓
📊 Visualizations
      ↓
📄 Final Responsible AI Report
```

---

# ✨ Key Features

### 📂 Dataset Processing

* Upload and process CSV datasets
* Automatic dataset validation
* Dataset statistics and summaries
* Missing-value detection
* Duplicate detection
* Categorical feature detection
* Automatic target-column detection
* Classification and regression task detection

### 🧹 Data Preprocessing

* Missing-value handling
* Duplicate removal
* Categorical encoding
* Numerical preprocessing
* Outlier handling
* Skewness correction
* Feature preprocessing

### ⚖️ Class Imbalance Handling

* Automatic imbalance detection
* Class distribution analysis
* SMOTE-based balancing
* SMOTENC support for categorical features
* Before/after imbalance visualization

### 🧬 Synthetic Data Generation

* CTGAN-based tabular data generation
* Synthetic dataset creation
* Real vs synthetic data comparison
* Synthetic data visualization

### 🔍 Bias & Fairness

* Bias detection
* Fairness analysis
* Protected attribute analysis
* Proxy bias detection
* Fairness comparison before and after mitigation
* Bias visualization
* Fairness improvement workflow

### 🤖 Machine Learning

Supported model families include:

* 🌲 Random Forest
* 📈 Gradient Boosting
* 🗳️ Voting-based models

The pipeline supports:

* Classification
* Regression
* Model evaluation
* Feature importance analysis
* Model saving

### 🔎 Explainable AI

The pipeline provides explainability-related analysis using:

* SHAP
* LIME
* Feature importance
* LLM-generated explanations

### 🧠 Generative AI

The project integrates an LLM-powered decision engine for:

* Dataset analysis
* Bias analysis
* Fairness interpretation
* Risk assessment
* Model analysis
* Responsible AI recommendations
* Optimization recommendations

### 📊 Responsible AI Metrics

The pipeline calculates multiple Responsible AI metrics:

| Metric                  | Description                           |
| ----------------------- | ------------------------------------- |
| 📊 EDQS                 | Ethical Data Quality Score            |
| ⚠️ ERI                  | Ethical Risk Index                    |
| 🤖 RAI                  | Responsible AI Index                  |
| ⚖️ Fairness Score       | Measures fairness-related performance |
| 🔎 Explainability Score | Measures model explainability         |
| 🏛️ Governance Score    | Overall governance assessment         |

### 📄 Automated Reporting

The pipeline automatically generates:

* 📊 Metric reports
* 📈 Graphs
* 🧠 LLM analysis
* 💡 Recommendations
* 📄 PDF Responsible AI report
* 💾 Final processed dataset
* 🤖 Trained model

---

# 🛠️ Tech Stack

## 🐍 Programming

* Python
* Pandas
* NumPy

## 🤖 Machine Learning

* Scikit-learn
* Random Forest
* Gradient Boosting
* Imbalanced-learn

## ⚖️ Responsible AI / Fairness

* Fairlearn
* AIF360

## 🧬 Synthetic Data

* CTGAN
* SDV
* Faker

## 🔎 Explainable AI

* SHAP
* LIME

## 🧠 Generative AI

* LangChain
* Groq API
* LLM-powered decision engine

## 📊 Visualization

* Matplotlib
* Seaborn
* Plotly

## 📄 Reporting

* ReportLab
* FPDF

## 🖥️ Interface

* Tkinter
* Streamlit

---

# 📁 Project Structure

```text
Responsible-AI-Pipeline/
│
├── 📄 main.py
├── 📄 config.py
├── 📄 requirements.txt
├── 📄 README.md
├── 📄 .gitignore
├── 🔐 .env
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
└── 📁 outputs/
    └── Generated files
```

---

# ⚙️ Installation

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
```

Navigate into the project:

```bash
cd YOUR_REPOSITORY
```

---

## 2️⃣ Create Virtual Environment

### 🪟 Windows

```powershell
python -m venv venv
```

Activate:

```powershell
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then activate again:

```powershell
.\venv\Scripts\Activate.ps1
```

### 🐧 Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

---

# 📦 Install Dependencies

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Install project dependencies:

```bash
pip install -r requirements.txt
```

---

# 🔐 Environment Variables

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_api_key_here
```

The project uses `python-dotenv` to load environment variables.

### ⚠️ Important

Never upload your `.env` file to GitHub.

Your `.gitignore` should contain:

```gitignore
.env
venv/
.venv/
__pycache__/
*.pyc
outputs/
*.pkl
```

🔒 **Never expose API keys, passwords, database credentials, or other secrets in the repository.**

---

# ▶️ Running the Project

After activating the virtual environment and installing dependencies:

```bash
python main.py
```

The application will start the Responsible AI pipeline.

You can select your CSV dataset through the file-selection interface.

If the GUI file picker is unavailable, the application can request the dataset path through the terminal.

---

# 📂 Input Dataset

The main pipeline accepts **CSV datasets**.

Example:

```text
dataset.csv
```

The pipeline automatically analyzes the dataset and attempts to detect:

* 🎯 Target column
* 📊 Task type
* 🔤 Categorical columns
* 🔢 Numerical columns
* ⚖️ Class imbalance
* ❌ Missing values
* 📑 Duplicate records

Supported ML task types:

```text
✅ Classification
✅ Regression
```

---

# 🔄 Pipeline Workflow

## 1️⃣ Dataset Loading

The dataset is loaded using Pandas.

The pipeline collects information including:

* Number of rows
* Number of columns
* Column names
* Data types
* Dataset statistics

---

## 2️⃣ 🔍 Dataset Validation

The validation stage checks:

* Missing values
* Duplicate records
* Data types
* Invalid values
* Dataset consistency
* Class distribution

---

## 3️⃣ 🧹 Data Preprocessing

The preprocessing stage can perform:

* Missing-value handling
* Duplicate removal
* Categorical encoding
* Outlier handling
* Skewness correction
* Feature preprocessing

---

## 4️⃣ ⚖️ Imbalance Detection

The pipeline identifies class imbalance in classification datasets.

It analyzes class distributions and determines whether balancing techniques are required.

---

## 5️⃣ 🔄 SMOTE Balancing

For classification datasets, the pipeline can use:

```text
SMOTE
```

and where categorical variables require special handling:

```text
SMOTENC
```

This helps reduce class imbalance before model training.

---

## 6️⃣ 🧬 CTGAN Synthetic Data

CTGAN can be used to generate synthetic tabular data.

The generated data can be compared against the original dataset using visualizations and statistical analysis.

---

## 7️⃣ 🔍 Bias Detection

The pipeline analyzes potential bias across relevant attributes.

It can generate:

* Bias metrics
* Fairness metrics
* Bias tables
* Bias visualizations
* Fairness comparisons
* Proxy-bias analysis

---

## 8️⃣ ⚖️ Fairness Improvement

The pipeline can apply fairness-related processing to improve model outcomes.

The results are compared before and after mitigation.

```text
Before Fairness Processing
          ↓
     Bias Analysis
          ↓
   Fairness Mitigation
          ↓
After Fairness Processing
```

---

## 9️⃣ 🤖 Model Training

The pipeline trains machine-learning models such as:

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

Multiple models can also be combined using voting-based approaches.

---

# 📊 Model Evaluation

## Classification Metrics

For classification tasks, the pipeline evaluates:

* 🎯 Accuracy
* 🎯 Precision
* 🎯 Recall
* 🎯 F1 Score

## Regression Metrics

For regression tasks:

* 📉 MAE
* 📉 MSE
* 📉 RMSE
* 📈 R² Score

---

# 📈 Responsible AI Metrics

## 📊 EDQS — Ethical Data Quality Score

EDQS evaluates dataset quality based on factors such as:

* Missing values
* Duplicate records
* Class imbalance

Example weighting configuration:

```python
EDQS_WEIGHTS = {
    "missing": 0.4,
    "duplicate": 0.3,
    "imbalance": 0.3
}
```

---

## ⚠️ ERI — Ethical Risk Index

ERI provides an aggregated view of ethical/model risk using relevant factors such as:

* Fairness
* Model performance
* Imbalance
* Explainability

---

## 🤖 RAI — Responsible AI Index

RAI combines multiple Responsible AI dimensions to provide an overall Responsible AI assessment.

It considers areas such as:

* Data quality
* Fairness
* Model performance
* Ethical risk

---

## 🔎 Explainability Score

The Explainability Score evaluates the explainability-related characteristics of the trained model.

The project uses techniques such as:

* SHAP
* LIME
* Feature importance

---

## 🏛️ Governance Score

The Governance Score provides an aggregated assessment of important Responsible AI dimensions.

It can be used to summarize:

* Data quality
* Fairness
* Risk
* Explainability
* Model performance

---

# 🧠 LLM Decision Engine

The project integrates a **Groq-powered LLM Decision Engine**.

The LLM analyzes the pipeline results and generates:

### 🔍 Analysis

* Dataset quality analysis
* Bias analysis
* Fairness analysis
* Ethical risk analysis
* Explainability analysis
* Model performance analysis

### 💡 Recommendations

The LLM can generate recommendations for improving:

* Data quality
* Fairness
* Model performance
* Explainability
* Responsible AI metrics

### 🔄 Optimization

Recommendations can be passed back into the pipeline's optimization workflow.

---

# ⚡ Optimization Configuration

Important configuration parameters include:

```python
FAST_MODE = True
TEST_SIZE = 0.20
RANDOM_STATE = 42
MAX_ITERATIONS = 5
PATIENCE = 2
MIN_IMPROVEMENT = 0.001
```

These parameters control:

* ⚡ Fast execution mode
* 📊 Train/test split
* 🎲 Reproducibility
* 🔁 Maximum optimization iterations
* ⏹️ Early stopping
* 📈 Minimum improvement threshold

---

# 📊 Visualizations

The pipeline can generate visualizations including:

* 📊 Before/after comparisons
* ⚖️ Bias comparisons
* 🔥 Bias heatmaps
* 🕸️ Bias radar charts
* 📈 EDQS comparison
* 📈 ERI comparison
* 📈 RAI comparison
* ⚖️ Fairness comparison
* 🔥 Correlation heatmaps
* 🌲 Feature importance
* 🧬 Real vs synthetic CTGAN data
* 📦 Boxplots
* 🥧 Pie charts
* 📊 Class distributions

---

# 📁 Generated Outputs

Generated files are stored inside:

```text
outputs/
```

Typical output structure:

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

The exact generated files depend on the selected dataset and task type.

---

# 📄 Automated PDF Report

The pipeline generates a PDF report containing important Responsible AI results.

The report can include:

* 📂 Dataset information
* 📊 Data quality metrics
* ⚖️ Fairness metrics
* ⚠️ Ethical Risk Index
* 🤖 Responsible AI Index
* 🔎 Explainability Score
* 🏛️ Governance Score
* 🎯 Model performance
* 🔍 Bias analysis
* 🧠 LLM-generated analysis
* 💡 LLM recommendations
* 📈 Before/after comparisons

---

# 🖥️ Streamlit Dashboard

A Streamlit dashboard is available at:

```text
modules/dashboard/app.py
```

Run the dashboard using:

```bash
streamlit run modules/dashboard/app.py
```

If the dashboard depends on generated outputs, run the main pipeline first:

```bash
python main.py
```

Then:

```bash
streamlit run modules/dashboard/app.py
```

---

# ⚙️ Configuration

Project-level configuration is maintained in:

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

# 🔐 Security

This project uses an API key for LLM functionality.

### ❌ Never commit:

```text
.env
API keys
Passwords
Database credentials
Private certificates
Secret tokens
```

### ✅ Before committing:

```bash
git status
```

Check that `.env` is not included.

You can also verify tracked files:

```bash
git ls-files
```

If an API key has accidentally been pushed to GitHub, **immediately revoke/rotate the key** and remove the secret from Git history.

---

# 🧪 Example Workflow

```text
📂 Select CSV Dataset
        ↓
🔍 Validate Dataset
        ↓
🧹 Clean & Preprocess
        ↓
⚖️ Detect Imbalance
        ↓
🔄 Apply SMOTE / SMOTENC
        ↓
🧬 Generate Synthetic Data
        ↓
🔍 Detect Bias
        ↓
⚖️ Apply Fairness Improvements
        ↓
🤖 Train ML Models
        ↓
📊 Evaluate Models
        ↓
🔎 Calculate Explainability
        ↓
📈 Calculate EDQS / ERI / RAI
        ↓
🏛️ Calculate Governance Score
        ↓
🧠 LLM Analysis
        ↓
💡 Generate Recommendations
        ↓
🔄 Optimize Pipeline
        ↓
📊 Generate Visualizations
        ↓
📄 Generate PDF Report
        ↓
💾 Save Final Dataset & Model
```

---

# 📌 Responsible AI Considerations

This project is designed as an engineering and research framework for analyzing Responsible AI characteristics in tabular machine-learning workflows.

The generated scores and recommendations should be treated as **decision-support outputs** rather than absolute judgments.

Responsible AI results can depend on:

* 📂 Dataset characteristics
* 🎯 Target definition
* ⚖️ Protected attributes
* 📊 Sampling strategy
* 🤖 Model selection
* 📈 Evaluation metrics
* ⚙️ Thresholds
* 🧹 Preprocessing techniques

Therefore, Responsible AI results should be interpreted within the context of the specific dataset and application.

---

# 🚀 Future Improvements

Possible future enhancements include:

* 🌐 Web-based dataset upload
* 📊 Interactive Responsible AI dashboard
* 🔐 Authentication and role-based access
* 🧠 More LLM providers
* ⚖️ Additional fairness algorithms
* 🤖 More machine-learning models
* 🔎 Interactive SHAP visualizations
* 📋 Automated Model Cards
* 🏛️ AI Governance audit trails
* 📦 Docker deployment
* ☁️ Cloud deployment
* 🔄 Experiment tracking
* 📡 REST API for pipeline execution
* 📊 Dataset versioning
* 📝 Automated compliance reports

---

# 🧑‍💻 Author

## Shahwaz Azam

🎓 B.Tech — Computer Science & Engineering (Data Science)

💡 Interested in:

* 🤖 Artificial Intelligence
* 🧠 Generative AI
* 🔎 Responsible AI
* 💻 Full Stack Development
* 📊 Machine Learning
* 🔗 RAG Systems

---

# ⭐ Support

If you find this project useful:

⭐ **Star the repository**

🍴 **Fork the repository**

🐛 **Open an issue**

💡 **Suggest improvements**

---

# 📜 License

This project is currently intended for **educational, research, and portfolio purposes**.

If you plan to distribute or use this project commercially, add an appropriate open-source license such as **MIT License**.

---

## 🚀 Quick Start

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git

# Enter project
cd YOUR_REPOSITORY

# Create environment
python -m venv venv

# Activate - Windows
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Configure API key
# Create .env and add GROQ_API_KEY

# Run pipeline
python main.py

# Optional: Run dashboard
streamlit run modules/dashboard/app.py
```

---

### 🤖 Built with Python • Machine Learning • Generative AI • Responsible AI • Explainable AI
