# ❤️ Heart Disease Classification - MLOps Project

This project implements a complete **MLOps pipeline** for heart disease classification, including data ingestion, preprocessing, versioning with Weights & Biases (W&B), and preparation for model training using PyTorch.

---

## 🚀 Project Overview

The goal of this project is to build a reproducible and scalable machine learning pipeline following MLOps best practices.

The pipeline includes:

- 📥 Automatic dataset download from Kaggle
- 🧹 Data cleaning (duplicates, outliers, skew handling)
- 🔢 Feature engineering (encoding + scaling)
- 📊 Data versioning with W&B (raw and processed)
- ⚙️ Modular pipeline structure

---

## 📊 Dataset

- **Source:** Kaggle  
- **Dataset:** Heart Disease Classification  
- **Records:** ~300 samples  
- **Features:** 14 original variables (medical attributes)

---

## 🧹 Data Preprocessing

### ✔ Data Cleaning
- Duplicate removal  
- Outlier handling (IQR method)  
- Skew correction (log transformation)  

### ✔ Feature Engineering
- One-Hot Encoding for categorical variables  
- Standardization using `StandardScaler`  

---

## 🔁 Experiment Tracking (W&B)

This project uses **Weights & Biases (W&B)** for:

- 📦 Dataset versioning  
- 📊 Metadata tracking  
- 🔁 Reproducibility  

### 📌 Logged Artifacts

- `raw_data` → original dataset  
- `processed_data` → cleaned and transformed dataset  

---

## ▶️ How to Run

### 1. Clone the repository

```bash
git clone https://github.com/SEU-USUARIO/heart-disease-classification-mlops.git
cd heart-disease-classification-mlops
```

### 2. Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate   # Windows
source venv/bin/activate # No Linux/Mac:
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Login to W&B

```bash
wandb login
```

### 4. Run the pipeline

```bash
python -m pipelines.run_pipeline
```

## 🧠 Technologies Used

- Python  
- Pandas & NumPy  
- Scikit-learn  
- PyTorch (planned for modeling)  
- Weights & Biases (W&B)  
- KaggleHub  

---

## 📌 Future Improvements

- Train/test split  
- Model training with PyTorch  
- Hyperparameter tuning  
- Pipeline automation  
- CI/CD integration  

---

## 👨‍💻 Author

Leonardo Dantas  

---

## 📜 License

This project is for educational purposes.
