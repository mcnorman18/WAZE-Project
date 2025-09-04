
# Waze Churn Prediction Project (Course 2 – Data Inspection)

**Prepared by:** Matthew Norman  

---

## Objective  
Inspect and analyze Waze’s churn dataset to prepare it for exploratory data analysis (EDA) and eventual machine learning model development.  
This project demonstrates the ability to clean, inspect, engineer features, and extract business insights using Python.  

---

## Approach, Code, and Thought Process  

### 1. Dataframe Creation & First Pass  
**Why:** Sanity-check the file loaded correctly, understand structure/size, and surface obvious problems early.  
**Questions:** What’s the row/column count? Which columns are numeric vs. categorical? Any glaring nulls?  

```python
import pandas as pd
import numpy as np

df = pd.read_csv('waze_dataset.csv')
df.head(10)
df.info()
```  

**Result / Thought Process:** Confirmed file integrity, row/column counts, and presence of missing values.  
If needed, I would convert dates with `pd.to_datetime` and make categorical columns explicit.  

---

### 2. Missingness Scan  
**Why:** Missing labels can bias analysis. Needed to see if missingness was random or patterned.  
**Questions:** Are rows with nulls systematically different? Any device skew?  

```python
df_nulls = df[df.isnull().any(axis=1)]
df_nonulls = df.dropna()

df_nulls.describe()
df_nonulls.describe()

df_nulls['device'].value_counts(normalize=True) * 100
df['device'].value_counts(normalize=True) * 100
```  

**Result / Thought Process:** About 700 rows missing churn labels. Distributions were consistent with non-missing rows, suggesting randomness. No device skew → missingness not likely a problem.  

---

### 3. Class Balance  
**Why:** Understand churn prevalence and anticipate modeling tactics.  
**Questions:** How imbalanced is the target?  

```python
df['label'].value_counts(normalize=True) * 100
```  

**Result / Thought Process:** ~82% retained, 18% churned. Moderate imbalance, so I’d plan to use stratified sampling and precision-focused metrics for modeling.  

---

### 4. Medians vs. Outliers  
**Why:** Outliers (e.g., 21,000 km in a month) can distort means. Medians show typical users.  
**Questions:** Do churned and retained users behave differently?  

```python
df.groupby('label').median(numeric_only=True)
```  

**Result / Thought Process:**  
- Churned users = more drives in fewer days, longer distances, more hours.  
- Retained users = steadier driving spread across more days.  
- Suggests churn is linked to intensity of driving behavior.  

---

### 5. Feature Engineering  
**Why:** Ratios often capture behavior better than raw counts.  
**Questions:** How far per drive? How concentrated is activity per day?  

```python
df['km_per_drive'] = df['driven_km_drives'] / df['drives']
df['km_per_driving_day'] = df['driven_km_drives'] / df['driving_days']
df['drives_per_driving_day'] = df['drives'] / df['driving_days']

df.groupby('label').median()[['km_per_drive','km_per_driving_day','drives_per_driving_day']]
```  

**Result / Thought Process:** Churned users were “super-drivers” with higher distances and more drives per day.  
These engineered features provide valuable signals for churn prediction.  

---

### 6. Device Neutrality Check  
**Why:** Validate whether device type influences churn.  
**Questions:** Does churn vary between iPhone and Android users?  

```python
df.groupby('label')['device'].value_counts(normalize=True) * 100
```  

**Result / Thought Process:** Device ratios were consistent across churned and retained groups.  
Device type is not a churn driver.  

---

## Key Takeaways  
- Missing values are random → dataset is suitable for modeling.  
- Device type does not influence churn.  
- Churned users are higher-intensity drivers; retained users drive more consistently.  
- Engineered features (km per drive, drives per day) provide valuable predictors for churn.  

---

## Value to Recruiters  
This project demonstrates my ability to:  
- Inspect and clean real-world datasets  
- Apply **Python (pandas, numpy)** for wrangling and feature creation  
- Use statistical reasoning (medians vs. means) to manage outliers  
- Translate raw code outputs into **business insights** that inform strategy  
