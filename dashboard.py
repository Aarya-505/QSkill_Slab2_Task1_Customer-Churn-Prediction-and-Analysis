import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Customer Churn Dashboard", layout="wide")

st.title("📊 Customer Churn Analysis Dashboard")
st.markdown("This dashboard provides exploratory data analysis and insights for the QSkill Virtual Internship Task 1.")

@st.cache_data
def load_data():
    df = pd.read_csv('Telco-Customer-Churn.csv')
    df['TotalCharges'] = df['TotalCharges'].replace(' ', 0).astype(float)
    return df

try:
    df = load_data()
    
    # Top Level Metrics
    total_customers = len(df)
    churn_rate = (df['Churn'] == 'Yes').mean() * 100
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Customers", f"{total_customers:,}")
    col2.metric("Overall Churn Rate", f"{churn_rate:.1f}%")
    col3.metric("Avg Monthly Charge", f"${df['MonthlyCharges'].mean():.2f}")
    
    st.divider()
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Churn Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(data=df, x='Churn', ax=ax, palette="Set2")
        st.pyplot(fig)
        
    with col2:
        st.subheader("Churn by Contract Type")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(data=df, x='Contract', hue='Churn', ax=ax, palette="Set1")
        st.pyplot(fig)
        
    st.divider()
    
    st.subheader("Key Findings & Recommendations")
    st.markdown("""
    * **Tenure:** Newer customers have a significantly higher risk of churning.
    * **Contracts:** Month-to-month contracts are highly correlated with churn. 
    * **Action Item:** Incentivize users to sign 1-year or 2-year contracts by offering discounts.
    """)
    
    st.dataframe(df.head())
except Exception as e:
    st.error(f"Error loading data: {e}. Please ensure Telco-Customer-Churn.csv is in the directory.")
