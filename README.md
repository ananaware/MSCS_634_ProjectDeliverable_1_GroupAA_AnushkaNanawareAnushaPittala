# MSCS 634 Project Deliverable 1: Data Collection, Cleaning, and Exploration

**Course:** MSCS-634-M50 Advanced Big Data and Data Mining (Fall 2026)
**Group AA:** Anushka Nanaware and Anusha Pittala
**Topic:** Predicting customer churn for a telecom company

## Files in this repository
- `MSCS_634_ProjectDeliverable_1.ipynb`: the full notebook (loading, inspection, cleaning, EDA and insights) with comments on all code
- `README.md`: this summary

## 1. Dataset Summary
- **Dataset:** Telco Customer Churn (IBM sample dataset)
- **Source:** https://github.com/IBM/telco-customer-churn-on-icp4d (file `Telco-Customer-Churn.csv`)
- **Size:** 7,043 customers (rows) and 21 attributes (columns)
- **What each row is:** one telecom customer, with their account details, the services they use, their billing, and whether they left the company (`Churn`)
- **Attribute types:**
  - Customer info: `gender`, `SeniorCitizen`, `Partner`, `Dependents`
  - Account info: `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod`
  - Services: `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`
  - Charges: `MonthlyCharges`, `TotalCharges`
  - Target: `Churn` (Yes/No)

**Why we chose it:** It is well above the 500-row and 8-attribute minimum, and it mixes numeric and categorical columns, so it supports every later deliverable: regression (charges), classification (predicting churn), clustering (customer groups) and association rules (the Yes/No service columns work as items). Churn is also a real business problem, since keeping a customer costs less than finding a new one.

## 2. Major Cleaning Steps
Cleaning was done on a copy (`df_clean`) so the raw data stays unchanged.

| Issue found | What we did | Why |
|---|---|---|
| `TotalCharges` stored as text, with 11 hidden blank values that `isnull()` missed | Converted to numbers and filled the 11 blanks with 0 | All 11 customers have tenure = 0 and have not churned, so they are new customers who were not billed yet. We kept the rows instead of deleting real customers. |
| Duplicates | Checked duplicate rows and duplicate customer IDs | None found (0 and 0) |
| Inconsistent labels: "No internet service" (6 columns) and "No phone service" (`MultipleLines`) | Changed them to "No" | The counts match exactly (1,526 and 682) with customers who have no internet or phone, so the labels repeat information already in other columns. |
| `SeniorCitizen` used 0/1 while other columns use Yes/No | Changed it to Yes/No | Consistent labels |
| Extra spaces in text | Checked all text columns | None found |
| Noisy data | Compared `TotalCharges` with tenure x `MonthlyCharges` | Average ratio is 1.0. Only 59 customers differ by more than 20% and only 2 by more than 50%, which is normal noise from price changes. Kept. |
| Outliers | IQR rule on `tenure`, `MonthlyCharges`, `TotalCharges` | 0 outliers in the full data. Kept all rows. |

Result: shape stays (7043, 21) with 0 missing values.

## 3. Exploration and Key Insights
We used Seaborn and Matplotlib for the churn balance, histograms, boxplots, churn rate by category, add-on services, a correlation heatmap and a scatter plot.

- **Churn is imbalanced:** 73.5% stayed and 26.5% churned.
- **Contract is the strongest signal:** month-to-month customers churn at 42.7%, one-year at 11.3% and two-year at only 2.8%.
- **New customers leave most:** the median tenure is 10 months for churners and 38 months for loyal customers.
- **Higher bills, more churn:** churners have a higher median monthly charge (79.6 vs 64.4). Fiber optic customers churn at 41.9% vs 19.0% for DSL.
- **Support services help keep customers:** customers with OnlineSecurity or TechSupport churn at about 15%, about half the rate of those without them (about 31%).
- **Payment method:** electronic check users churn at 45.3%, much higher than automatic payments (15% to 17%).
- **Gender does not matter** (26.9% vs 26.2%), while senior citizens churn more (41.7% vs 23.6%).
- **Correlations:** `TotalCharges` is strongly linked to `tenure` (0.83) and `MonthlyCharges` (0.65). Churn has a moderate negative link with tenure (-0.35).

**How this guides future modeling:** we will use stratified splits and precision, recall and F1 for classification because of the class imbalance. For regression, `MonthlyCharges` is a better target than `TotalCharges` (which is almost tenure x `MonthlyCharges`), and Ridge or Lasso can handle the overlap between features. We plan features such as tenure bands, a count of add-on services and a log of `TotalCharges`. We will also check results across gender and senior-citizen groups for fairness.

## 4. Challenges and How We Handled Them
- **Hidden missing values:** `isnull()` reported 0 missing, but `TotalCharges` had 11 blank strings. We checked the data types, found the text column, converted it, and used the tenure = 0 pattern to decide that 0 was the right fill.
- **Repeated labels:** "No internet service" and "No phone service" would have created extra categories with no new information. We proved they match the internet/phone columns and merged them into "No".
- **Deciding what is noise or an outlier:** Nothing was an outlier by the IQR rule. In boxplots split by Churn, a few long-tenure churners (23) and high-total churners (109) appear as dots, but they are real customers, so we kept them.
- **Imbalanced target:** only 26.5% of customers churn. We noted this early so later models use suitable metrics and methods.
- **Correlated features:** `TotalCharges` overlaps with `tenure` and `MonthlyCharges`. We flagged it for the regression deliverable.

## How to run
1. Open the notebook in Google Colab (or Jupyter).
2. Run all cells from the top. The data loads directly from the IBM GitHub link, so no file upload is needed.
3. Libraries used: pandas, numpy, matplotlib, seaborn.
