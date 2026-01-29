import pandas as pd
from sklearn.preprocessing import LabelEncoder

# 1. Load the dataset
print("⏳ Processing data, please wait...")
df = pd.read_csv('BankChurners.csv')

# 2. Drop unnecessary columns
# Removing the last 2 garbage columns from Kaggle and the CLIENTNUM (ID) column.
# CLIENTNUM is irrelevant for prediction and should be removed.
df = df.iloc[:, :-2]
if 'CLIENTNUM' in df.columns:
    df = df.drop('CLIENTNUM', axis=1)

# 3. Encode Target Variable (Churn)
# Existing Customer -> 0
# Attrited Customer -> 1 (Target)
df['Attrition_Flag'] = df['Attrition_Flag'].apply(lambda x: 1 if x == 'Attrited Customer' else 0)

# ---------------------------------------------------------
# 4. SMART ENCODING (PROFESSIONAL TOUCH)
# ---------------------------------------------------------

# A) Ordinal Data (Order matters!)
# We manually map these features because there is a logical hierarchy.
# (e.g., Doctorate > High School, Platinum > Blue)

education_order = {
    'Unknown': 0, 'Uneducated': 1, 'High School': 2, 'College': 3,
    'Graduate': 4, 'Post-Graduate': 5, 'Doctorate': 6
}
income_order = {
    'Unknown': 0, 'Less than $40K': 1, '$40K - $60K': 2, '$60K - $80K': 3,
    '$80K - $120K': 4, '$120K +': 5
}
card_order = {
    'Blue': 1, 'Silver': 2, 'Gold': 3, 'Platinum': 4
}

# Apply mappings
df['Education_Level'] = df['Education_Level'].map(education_order)
df['Income_Category'] = df['Income_Category'].map(income_order)
df['Card_Category'] = df['Card_Category'].map(card_order)

print("✅ Smart Encoding: Education, Income, and Card Category mapped successfully.")

# B) Nominal Data (No inherent order)
# For Gender and Marital Status, standard Label Encoding is sufficient.
categorical_cols = ['Gender', 'Marital_Status']
le = LabelEncoder()

for col in categorical_cols:
    df[col] = le.fit_transform(df[col])
    print(f"✅ Standard Encoding: {col}")

# 5. Save the Cleaned Data
df.to_csv('cleaned_data.csv', index=False)

print("\n🎉 SUCCESS! Data cleaning completed.")
print("📂 New file created: 'cleaned_data.csv'")
print("-" * 30)
print(df.head())