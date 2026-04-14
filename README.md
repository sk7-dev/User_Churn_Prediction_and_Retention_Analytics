# Waze User Churn Analysis and Prediction

## Overview

This project analyzes user behavior data from Waze to understand churn patterns and build predictive models that can help identify users at risk of leaving the platform. The work moves through a full analytics and machine learning workflow: exploratory data analysis, hypothesis testing, feature engineering, baseline modeling, and advanced model development.

The project uses behavioral usage data to answer two practical business questions:

1. **What user behaviors are associated with churn?**
2. **Can churn be predicted accurately enough to support retention efforts?**

Across the notebooks, the analysis shows that churn is driven much more by **behavioral intensity and usage patterns** than by **device type**. Users who churn tend to drive more intensely within fewer active days, while retained users tend to engage more consistently over time. A baseline logistic regression model established an initial benchmark, and tree-based models improved recall for churn detection. Threshold tuning further increased the model’s usefulness for retention campaigns where false positives are relatively low cost.

---

## Business Problem

Waze wants to better understand which users are likely to churn so the company can design more targeted retention strategies. Since sending a reminder, in-app message, or promotional outreach is relatively low risk, even a model with moderate precision can still be valuable if it improves recall and helps identify a larger share of at-risk users.

This project supports that goal by:
- Profiling churned vs. retained users
- Testing whether device differences matter
- Engineering behavior-based features
- Building classification models to predict churn
- Evaluating whether threshold adjustments can better align model behavior with business needs

---

## Dataset

The project uses a Waze dataset with:

- **14,999 total user records**
- **13 original features**
- **700 missing values in the `label` column**
- A target distribution of approximately:
  - **82.3% retained**
  - **17.7% churned**

### Core Variables
The dataset includes usage and driving behavior features such as:

- `sessions`
- `drives`
- `total_sessions`
- `n_days_after_onboarding`
- `total_navigations_fav1`
- `total_navigations_fav2`
- `driven_km_drives`
- `duration_minutes_drives`
- `activity_days`
- `driving_days`
- `device`
- `label`

---

## Project Structure

```text
.
├── eda.ipynb
├── hypothesis_testing.ipynb
├── baseline_model.ipynb
├── model.ipynb
└── README.md
```

### Notebook Summary

#### `eda.ipynb`
Performs initial data inspection, null analysis, descriptive statistics, churn segmentation, and exploratory feature comparisons. This notebook identifies several meaningful behavioral differences between churned and retained users.

#### `hypothesis_testing.ipynb`
Tests whether average driving activity differs between iPhone and Android users using a two-sample t-test.

#### `baseline_model.ipynb`
Builds a logistic regression baseline using engineered features and evaluates its performance on churn classification.

#### `model.ipynb`
Builds and evaluates more advanced machine learning models, including Random Forest and XGBoost, then explores threshold tuning to improve recall.

---

## Tools and Libraries

The notebooks use Python and standard data science libraries, including:

- **Python**
- **pandas**
- **numpy**
- **matplotlib**
- **seaborn** / visualization helpers where applicable
- **scipy**
- **scikit-learn**
- **xgboost**

---

## Workflow

## 1. Exploratory Data Analysis

The analysis started by reviewing structure, data types, completeness, and churn distribution.

### Key EDA findings

- The dataset contains **14,999 rows** and **13 columns**
- The `label` column has **700 missing values**
- Missing-label records appeared broadly similar to labeled records based on summary statistics
- The class distribution is moderately imbalanced:
  - **11,763 retained**
  - **2,536 churned**

### Churn behavior insights

Median comparisons revealed strong behavioral differences between churned and retained users:

- Median churned users had **50 drives**, compared with **47 drives** for retained users
- Median churned users were active on only **8 activity days**, while retained users were active on **17 activity days**
- Median churned users had **6 driving days**, while retained users had **14 driving days**
- Median churned users drove approximately **3,653 km**, versus **3,465 km** for retained users
- Median churned users spent more total time driving than retained users

These results suggest churned users tend to concentrate more driving activity into fewer days, indicating a more intense but less consistent usage pattern.

### Engineered behavior signals from EDA

Additional features highlighted a clearer churn profile:

- Median **kilometers per driving day**
  - Churned: **697.54**
  - Retained: **289.55**

- Median **drives per driving day**
  - Churned: **10.00**
  - Retained: **4.06**

This indicates churned users drive much more per active driving day and may represent a different type of user behavior than retained users.

### Device distribution insight

The ratio of iPhone vs. Android users was nearly identical across churned and retained groups, suggesting device type alone is unlikely to explain churn behavior.

---

## 2. Hypothesis Testing

The project tested the following question:

**Do iPhone users and Android users have the same average number of drives?**

### Hypotheses

- **Null hypothesis (H₀):** There is no difference in average drives between iPhone and Android users
- **Alternative hypothesis (H₁):** There is a difference in average drives between iPhone and Android users

### Test used
- Welch’s two-sample t-test

### Results

- Mean drives for iPhone users: **67.86**
- Mean drives for Android users: **66.23**
- Test statistic: **1.4635**
- p-value: **0.1434**

### Conclusion

Since the p-value is greater than 0.05, the null hypothesis was not rejected. There is **no statistically significant difference** in average drives between iPhone and Android users. This helped eliminate device type as a major explanatory variable for churn-related behavior.

---

## 3. Feature Engineering

Several new features were engineered to improve signal quality for classification.

### Engineered features used across modeling work

- `km_per_driving_day`
- `professional_driver`
- `percent_sessions_in_last_month`
- `total_sessions_per_day`
- `km_per_hour`
- `km_per_drive`
- `percent_of_drives_to_favorite`
- Encoded binary features for target and device values

### Notable engineered feature insight

A `professional_driver` feature was created using the rule:

- `drives >= 60`
- `driving_days >= 15`

This produced:

- **12,405 non-professional drivers**
- **2,594 professional drivers**

Churn rate by this segment:

- Non-professional drivers: **19.9% churn**
- Professional drivers: **7.6% churn**

This was one of the clearest segmentation signals in the project.

---

## 4. Baseline Model: Logistic Regression

A logistic regression model was built as the first predictive benchmark.

### Modeling notes

- The target variable was converted into a binary churn label
- Features were selected from both original and engineered variables
- The model was trained using a train/test split
- A confusion matrix and classification report were used for evaluation

### Logistic regression performance

On the test set:

- **Accuracy:** 0.8238
- **Precision (churn):** 0.5179
- **Recall (churn):** 0.0915
- **F1-score (churn):** 0.16

### Interpretation

The logistic regression model achieved strong overall accuracy, but accuracy was inflated by class imbalance. Its churn recall was low, meaning it failed to identify most users who actually churned.

The notebook conclusion also showed that:

- `activity_days` was the most influential feature
- More activity days were associated with lower churn likelihood
- `km_per_driving_day` also contributed meaningful predictive signal

This baseline established that churn prediction was possible, but that a linear model was not sufficient for strong churn capture.

---

## 5. Advanced Models: Random Forest and XGBoost

To improve churn detection, the project trained tree-based models using train, validation, and test splits.

### Random Forest

Cross-validation results:

- **Precision:** 0.4572
- **Recall:** 0.1268
- **F1:** 0.1984
- **Accuracy:** 0.8185

Validation results:

- **Precision:** 0.4453
- **Recall:** 0.1203
- **F1:** 0.1894
- **Accuracy:** 0.8175

### XGBoost

Cross-validation results:

- **Precision:** 0.4259
- **Recall:** 0.1708
- **F1:** 0.2437
- **Accuracy:** 0.8119

Validation results:

- **Precision:** 0.4227
- **Recall:** 0.1617
- **F1:** 0.2340
- **Accuracy:** 0.8122

Test results:

- **Precision:** 0.4240
- **Recall:** 0.1815
- **F1:** 0.2541
- **Accuracy:** 0.8112

### Interpretation

XGBoost outperformed both the logistic regression baseline and the Random Forest model on churn recall while maintaining similar accuracy and precision. Although recall remained modest, it nearly doubled the baseline model’s ability to identify churned users.

The notebook also notes that the model produced more false negatives than false positives, which matters because missed churners are usually more costly than unnecessary outreach.

---

## 6. Threshold Tuning

Because retention campaigns often tolerate false positives, the project explored decision-threshold tuning for XGBoost.

### Default threshold performance
At the default classification behavior:

- **Precision:** 0.4240
- **Recall:** 0.1815
- **F1:** 0.2541
- **Accuracy:** 0.8112

### Threshold = 0.4

- **Precision:** 0.4150
- **Recall:** 0.2840
- **F1:** 0.3372
- **Accuracy:** 0.8021

### Threshold = 0.194

- **Precision:** 0.2932
- **Recall:** 0.4990
- **F1:** 0.3693
- **Accuracy:** 0.6979

### Interpretation

Lowering the threshold substantially improved recall. At a threshold of **0.194**, the model identified roughly **half of churners**, at the cost of lower precision and accuracy. This tradeoff may be acceptable in low-cost retention settings where reaching more at-risk users is more valuable than minimizing false alarms.

---

## Key Findings

### 1. Device type is not a meaningful differentiator
Despite a slight difference in mean drives between iPhone and Android users, hypothesis testing showed the difference was not statistically significant.

### 2. Churn is more strongly associated with behavior intensity than platform
Users who churn tend to have:
- more drives packed into fewer days
- greater distance per active day
- lower sustained engagement across the month

### 3. Activity consistency matters
Features related to active usage days were among the strongest predictors in the baseline model. More consistent engagement appears associated with retention.

### 4. Feature engineering improved predictive value
Engineered features accounted for a large share of the most important predictors in the advanced model, highlighting the importance of transforming raw usage data into behavioral indicators.

### 5. Model usefulness depends on the business objective
The default model is not strong enough for high-stakes decisions, but threshold tuning makes it more useful for broad retention outreach where false positives are acceptable.

---

## Quantified Results

- Analyzed **14,999** user records with **700 missing churn labels**
- Established churn prevalence at **17.7%**
- Found churned users had median **10.0 drives/day** versus **4.06** for retained users
- Found churned users had median **697.54 km/day** versus **289.55 km/day**
- Identified a major segment effect:
  - **7.6% churn** for professional drivers
  - **19.9% churn** for non-professional drivers
- Rejected device-based targeting assumptions with:
  - **t = 1.4635**
  - **p = 0.1434**
- Improved churn recall from **0.0915** (logistic regression) to **0.1815** (XGBoost default)
- Further increased recall to **0.4990** through threshold tuning

---

## Business Recommendations

Based on the notebook results, the strongest next steps would be:

1. **Target retention efforts using behavioral features instead of device type**
   Device type was not statistically meaningful, while activity intensity and driving patterns were.

2. **Prioritize users with high intensity but low consistency**
   Users with many drives compressed into few driving days may be at higher churn risk.

3. **Use threshold-tuned predictions for low-cost outreach**
   If the intervention is inexpensive, a lower classification threshold can help capture many more churners.

4. **Expand feature development**
   Since engineered variables contributed heavily to performance, additional user-lifecycle and temporal features may further improve churn detection.

5. **Treat current model as directional, not definitive**
   The model is useful for experimentation and prioritization, but not yet strong enough for fully automated or high-stakes business decisions.

---
