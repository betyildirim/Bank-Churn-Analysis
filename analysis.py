import pandas as pd
import plotly.express as px
import numpy as np

# ---------------------------------------------------------
# 1. DATA LOADING & PREPARATION
# ---------------------------------------------------------
print("⏳ Loading dataset for analysis...")
df = pd.read_csv("BankChurners.csv")

# Remove last two unnecessary columns (Kaggle garbage)
df = df.iloc[:, :-2]

print("✅ Data loaded successfully!")
print(f"Total Customers: {df.shape[0]}")
print(f"Total Features: {df.shape[1]}")

# --- GLOBAL PROFESSIONAL COLOR PALETTE ---
# Consistent colors across all charts
colors = {
    "Existing Customer": "#1f77b4", # Professional Muted Blue
    "Attrited Customer": "#d62728"  # Professional Brick Red
}
order = ["Existing Customer", "Attrited Customer"]

# ---------------------------------------------------------
# CHART 1: CHURN DISTRIBUTION (PIE CHART)
# ---------------------------------------------------------
churn_counts = df["Attrition_Flag"].value_counts()

fig_pie = px.pie(
    names=churn_counts.index,
    values=churn_counts.values,
    title="Customer Churn Overview",
    color=churn_counts.index,
    color_discrete_map=colors # Uses professional colors
)

fig_pie.update_traces(textinfo="percent", textposition="inside", textfont_size=16)
fig_pie.update_layout(showlegend=True, legend_title_text="Customer Status")

fig_pie.write_html("01_churn_pie.html")
print("📊 Saved: 01_churn_pie.html")

# ---------------------------------------------------------
# CHART 2: CUSTOMER COUNT (BAR CHART) - UPDATED COLORS!
# ---------------------------------------------------------
bar_df = churn_counts.reset_index()
bar_df.columns = ["Attrition_Flag", "Count"]

# Ensure specific order
bar_df["Attrition_Flag"] = pd.Categorical(bar_df["Attrition_Flag"], categories=order, ordered=True)
bar_df = bar_df.sort_values("Attrition_Flag")

fig_bar = px.bar(
    bar_df,
    x="Attrition_Flag",
    y="Count",
    text="Count",
    title="Customer Count by Churn Status",
    color="Attrition_Flag",
    # ✅ FIXED: Now using the consistent 'colors' dictionary instead of basic "blue"/"red"
    color_discrete_map=colors 
)

fig_bar.update_traces(
    texttemplate="%{text:,}",   # Format: 8,500
    textposition="outside",
    marker_line_width=0,
    width=0.45
)

fig_bar.update_layout(
    xaxis_title="Customer Status",
    yaxis_title="Number of Customers",
    showlegend=False,
    yaxis_range=[0, 9000]
)

fig_bar.write_html("02_customer_count_by_churn.html")
print("📊 Saved: 02_customer_count_by_churn.html")

# ---------------------------------------------------------
# CHART 3: CHURN RATE BY INCOME (FIXED DOLLAR SIGNS)
# ---------------------------------------------------------
tmp = df.copy()

# Cleaning Income Strings
tmp["Income_Category"] = tmp["Income_Category"].astype(str).str.strip()
tmp["Income_Category"] = (
    tmp["Income_Category"]
      .str.replace(r"\s*-\s*", " - ", regex=True)
      .str.replace(r"\s*\+\s*", " +", regex=True)
      .str.replace(r"\s+", " ", regex=True)
)

# HTML Entity Trick for Dollar Signs
label_map = {
    "Less than $40K": "Less than &#36;40K",
    "$40K - $60K": "&#36;40K - &#36;60K",
    "$60K - $80K": "&#36;60K - &#36;80K",
    "$80K - $120K": "&#36;80K - &#36;120K",
    "$120K +": "&#36;120K +",
    "Unknown": "Unknown"
}
tmp["Income_Label"] = tmp["Income_Category"].map(label_map).fillna(tmp["Income_Category"])

# Calculate Churn Rate
tmp["is_churn"] = (tmp["Attrition_Flag"] == "Attrited Customer").astype(int)
income_rate = tmp.groupby("Income_Label", as_index=False)["is_churn"].mean()
income_rate["churn_rate_pct"] = (income_rate["is_churn"] * 100).round(2)

# Sorting
income_order = [
    "Less than &#36;40K", "&#36;40K - &#36;60K", "&#36;60K - &#36;80K", 
    "&#36;80K - &#36;120K", "&#36;120K +", "Unknown"
]
income_rate["Income_Label"] = pd.Categorical(income_rate["Income_Label"], categories=income_order, ordered=True)
income_rate = income_rate.sort_values("Income_Label")

fig_income = px.bar(
    income_rate,
    x="Income_Label",
    y="churn_rate_pct",
    text="churn_rate_pct",
    title="Churn Rate (%) by Income Category"
)

fig_income.update_traces(texttemplate="%{text:.2f}%", textposition="outside", cliponaxis=False)
fig_income.update_layout(
    xaxis_title="Income Category",
    yaxis_title="Churn Rate (%)",
    xaxis=dict(tickangle=0),
    margin=dict(b=50),
    yaxis=dict(range=[0, 25])
)

fig_income.write_html("03_churn_rate_by_income.html")
print("📊 Saved: 03_churn_rate_by_income.html")

# ---------------------------------------------------------
# CHART 4: TOTAL TRANSACTION BOX PLOT
# ---------------------------------------------------------
fig_tx = px.box(
    df,
    x="Attrition_Flag",
    y="Total_Trans_Ct",
    category_orders={"Attrition_Flag": order},
    color="Attrition_Flag",
    color_discrete_map=colors, # Uses professional colors
    points="suspectedoutliers",
    title="Distribution of Total Transactions by Customer Status"
)

fig_tx.update_layout(
    xaxis_title="Customer Status",
    yaxis_title="Total Transactions",
    showlegend=False
)

fig_tx.write_html("04_total_trans_ct_box.html")
print("📊 Saved: 04_total_trans_ct_box.html")

# ---------------------------------------------------------
# CHART 5: AGE DISTRIBUTION (NORMALIZED & OVERLAY)
# ---------------------------------------------------------
fig_age = px.histogram(
    df,
    x="Customer_Age",
    color="Attrition_Flag",
    color_discrete_map=colors, # Uses professional colors
    histnorm="percent",
    nbins=25,
    barmode="overlay",  # Overlay for comparison
    opacity=0.6,        # Transparency
    marginal="box",     # Top box plot
    template="plotly_white",
    title="Customer Age Distribution by Churn Status (Normalized)"
)

fig_age.update_layout(
    xaxis_title="Customer Age",
    yaxis_title="Percentage of Customers (%)",
    legend_title_text="Status",
    bargap=0.05
)

fig_age.update_traces(marker_line_width=1, marker_line_color="white")

fig_age.write_html("05_age_hist_by_churn.html")
print("📊 Saved: 05_age_hist_by_churn.html")

print("\n🎉 All Analysis Charts Generated Successfully!")