# Customer Churn Analysis & Recommendations

## 1. Project Overview
This project analyzes a real-world customer dataset (Telco Customer Churn) to identify the key factors causing customers to leave and predict potential churners using machine learning.

## 2. Model Evaluation Summary
We trained and evaluated two classification models after resolving class imbalance using SMOTE.

**Logistic Regression Performance:**
- **ROC-AUC Score:** 0.8240
- **Recall (Churn):** 0.72 (Successfully identifies 72% of actual churners)
- **Precision (Churn):** 0.50
- **F1-Score (Churn):** 0.59

**Random Forest Performance:**
- **ROC-AUC Score:** 0.8162
- **Recall (Churn):** 0.62 
- **Precision (Churn):** 0.54
- **F1-Score (Churn):** 0.58

*Conclusion:* Logistic Regression performed slightly better in identifying true churners (higher recall and ROC-AUC), making it the preferred model for a churn prediction use case where failing to detect a churning customer is more costly.

## 3. Top Factors Driving Churn
Based on our Random Forest feature importance analysis, the top 5 factors that dictate whether a customer stays or leaves are:
1. **Tenure:** How long they have been a customer. Newer customers are far more likely to churn.
2. **Total Charges:** Total amount billed over their lifetime. 
3. **Monthly Charges:** The monthly bill amount. High monthly charges correlate strongly with churn.
4. **Payment Method (Electronic Check):** Customers who pay manually via Electronic Check churn at much higher rates.
5. **Contract (Two Year):** Having a long-term contract heavily reduces the likelihood of churn.

## 4. Practical Business Recommendations
Based on the data and the identified churn factors, we recommend the following practical strategies to reduce customer churn:

* **Incentivize Long-Term Contracts:** 
  Customers on "Month-to-month" contracts are the most at risk. Offer discounts or premium feature upgrades to encourage customers to sign One-Year or Two-Year contracts.
* **Review Pricing & Monthly Charges:** 
  High monthly charges are a major driver of churn. Consider offering personalized discounts or loyalty bundles to customers whose monthly charges cross a specific risk threshold.
* **Target Newer Customers for Retention:** 
  Since low `tenure` is the #1 churn driver, the company needs a stronger onboarding process. Focus heavily on customer satisfaction check-ins and dedicated support during the first 6-12 months.
* **Encourage Automated Payments:** 
  Customers using "Electronic Checks" are churning heavily. Offer a small monthly discount ($5-$10) for customers who switch to automatic credit card payments or bank transfers.
