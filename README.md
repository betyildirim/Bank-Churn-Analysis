# 🏦 Bank Customer Churn Prediction

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Random%20Forest-green)
![Visualization](https://img.shields.io/badge/Visualization-Plotly-orange)

## 📌 Project Overview
This project aims to predict customer churn for a bank using Machine Learning. By analyzing customer demographics and transaction history, we identify the key factors that drive customers to leave. The solution includes data cleaning, exploratory data analysis (EDA), and a Random Forest classification model.

## 🚀 Key Features
- **Smart Data Preprocessing:** Handling ordinal and nominal categorical variables with custom mapping.
- **Exploratory Data Analysis (EDA):** Interactive visualizations to understand churn distribution, age, and transaction behaviors.
- **Machine Learning Model:** A Random Forest Classifier trained to predict churn with high accuracy.
- **Feature Importance Analysis:** Identifying the top drivers of churn (e.g., Total Transaction Amount, Transaction Count).

## 📂 File Structure
- `preprocessing.py` -> Cleans raw data and applies feature engineering.
- `analysis.py` -> Generates interactive Plotly charts for data insights.
- `train_model.py` -> Trains the ML model and visualizes feature importance.
- `cleaned_data.csv` -> The processed dataset used for training.

## 📊 Insights & Results
The analysis revealed that **customer engagement** is the biggest predictor of churn.
1.  **Total Transaction Amount:** Customers with lower transaction amounts are more likely to churn.
2.  **Transaction Count:** Active users tend to stay.
3.  **Revolving Balance:** Customers with higher revolving balances are less likely to leave.

## 🛠️ Technologies Used
- **Language:** Python
- **Libraries:** Pandas, Scikit-Learn, Plotly, NumPy

## 💻 How to Run
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/betyildirim/Bank-Churn-Analysis.git](https://github.com/betyildirim/Bank-Churn-Analysis.git)
    ```
2.  **Install dependencies:**
    ```bash
    pip install pandas plotly scikit-learn
    ```
3.  **Run the scripts in order:**
    ```bash
    python preprocessing.py
    python analysis.py
    python train_model.py
    ```

---
*Created by Betül*
