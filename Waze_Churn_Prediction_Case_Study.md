
# Waze Churn Prediction Project (Course 2 – Data Inspection)

## Objective
Inspect and analyze Waze’s churn dataset to prepare it for exploratory data analysis (EDA) and eventual machine learning model development.

---

## Approach & Code

### 1. Dataframe Creation
Loaded the churn dataset into a pandas dataframe to review its structure and data types.
```python
import pandas as pd
import numpy as np

df = pd.read_csv('waze_dataset.csv')
df.head(10)
df.info()
```

---

### 2. Missing Values
Identified ~700 rows with null values (mostly in the churn label). Compared populations with and without missing data.
```python
df[df.isnull().any(axis=1)].describe()
df.dropna().describe()
```
**Result:** No significant differences → missingness appears random.

---

### 3. Device Distribution
Counted device types for rows with nulls and compared to overall dataset.
```python
df[df.isnull().any(axis=1)]['device'].value_counts()
df['device'].value_counts(normalize=True) * 100
```
**Result:** iPhone ~64%, Android ~36% — consistent across missing/non-missing and churned/retained groups. Device type is not a churn driver.

---

### 4. Churn vs. Retained Users
Reviewed user distribution and compared behavioral medians to avoid outlier distortion.
```python
df['label'].value_counts(normalize=True) * 100
df.groupby('label').median(numeric_only=True)
```
**Result:** ~82% retained, 18% churned.  
- Churned users drove more in fewer days, with longer trips and more hours.  
- Retained users had steadier usage across more days.

---

### 5. Feature Engineering for Deeper Insights
Created new behavioral metrics for analysis.
```python
df['km_per_drive'] = df['driven_km_drives'] / df['drives']
df['km_per_driving_day'] = df['driven_km_drives'] / df['driving_days']
df['drives_per_driving_day'] = df['drives'] / df['driving_days']
df.groupby('label').median()[['km_per_drive','km_per_driving_day','drives_per_driving_day']]
```
**Result:** Churned users were “super-drivers” — higher distance and drive concentration per day.

---

## Key Takeaways
- Missing values are random; dataset is suitable for further modeling.  
- Device type does not influence churn.  
- Churned users differ from retained users by **intensity of usage**, not device.  
- Feature engineering (km per drive/day, drives per day) provides meaningful predictors for churn modeling.

---

## Value to Recruiters
This project demonstrates my ability to:
- Inspect and clean real-world datasets.  
- Apply **Python (pandas, numpy)** for data wrangling and feature creation.  
- Use statistical reasoning (medians vs. means) to handle outliers.  
- Translate technical outputs into **business insights** for churn reduction.  
