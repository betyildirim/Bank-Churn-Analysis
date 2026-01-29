import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# ---------------------------------------------------------
# 1. Model Training
# ---------------------------------------------------------
print("Processing...")
df = pd.read_csv('cleaned_data.csv')

X = df.drop(columns=['Attrition_Flag', 'CLIENTNUM'], errors='ignore')
y = df['Attrition_Flag']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, 'churn_model.pkl')
print("Model trained and saved.")

# ---------------------------------------------------------
# 2. Feature Importance Calculation
# ---------------------------------------------------------
feature_names = X.columns

importances = (
    pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_
    })
    .sort_values("importance", ascending=False)
    .head(15)
    .reset_index(drop=True)
)

# Business Label Mapping
rename_map = {
    "Total_Trans_Amt": "Total Transaction Amount",
    "Total_Trans_Ct": "Total Transaction Count",
    "Total_Ct_Chng_Q4_Q1": "Transaction Count Change (Q4 vs Q1)",
    "Total_Revolving_Bal": "Revolving Credit Balance",
    "Avg_Utilization_Ratio": "Avg. Card Utilization Ratio",
    "Total_Relationship_Count": "Total Number of Products",
    "Total_Amt_Chng_Q4_Q1": "Transaction Amount Change (Q4 vs Q1)",
    "Credit_Limit": "Credit Limit",
    "Customer_Age": "Customer Age",
    "Avg_Open_To_Buy": "Available Credit (Open to Buy)",
    "Months_Inactive_12_mon": "Months Inactive (Last 12 Mon)",
    "Contacts_Count_12_mon": "Contact Count (Last 12 Mon)",
    "Months_on_book": "Months on Book (Tenure)",
    "Dependent_count": "Dependent Count",
    "Education_Level": "Education Level"
}

importances["feature_clean"] = importances["feature"].map(rename_map).fillna(importances["feature"])
importances["rank"] = importances.index + 1
importances_plot = importances.sort_values("importance", ascending=True)

importances_plot["display_name"] = (
    importances_plot["rank"].astype(str) + ". " + importances_plot["feature_clean"]
)

# ---------------------------------------------------------
# 3. Visualization
# ---------------------------------------------------------
fig_imp = px.bar(
    importances_plot,
    x="importance",
    y="display_name",
    orientation="h",
    text="importance",
    title="Key Drivers of Customer Churn (Random Forest)",
    template="plotly_white",
    color="importance",
    color_continuous_scale="Blues"
)

fig_imp.update_traces(
    texttemplate="%{text:.3f}",
    textposition="outside",
    cliponaxis=False
)

fig_imp.update_layout(
    title_x=0.5,
    xaxis_title="Importance Score",
    yaxis_title="",
    width=1150,
    height=700,
    margin=dict(l=280, r=50, t=80, b=50),
    showlegend=False,
    coloraxis_showscale=False,
    yaxis=dict(ticksuffix="   ")  # Padding between labels and bars
)

fig_imp.write_html("06_feature_importance.html")
print("Plot saved: 06_feature_importance.html")
fig_imp.show()