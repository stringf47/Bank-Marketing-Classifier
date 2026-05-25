# Bank Marketing Term Deposit Prediction

**CMPT459: Data Mining, Fall 2024**  
Anagh Arya — 301416450

## Problem Definition

Predict whether a client will subscribe (**yes/no**) to a term deposit based on data from direct marketing campaigns (phone calls) conducted by a Portuguese banking institution.

**Dataset:** [UCI Bank Marketing Dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing)

---

## 1. Dataset

The dataset consists of **45,211 instances** and **16 features** (multivariate, used for classification).

| Name | Type | Description |
|---|---|---|
| age | Integer | Client age |
| job | Categorical | Type of job (admin, blue-collar, entrepreneur, etc.) |
| marital | Categorical | Marital status (divorced, married, single, unknown) |
| education | Categorical | Education level |
| default | Binary | Has credit in default? |
| balance | Integer | Average yearly balance |
| housing | Binary | Has housing loan? |
| loan | Binary | Has personal loan? |
| contact | Categorical | Contact communication type (cellular, telephone) |
| day_of_week | Date | Last contact day of the week |
| month | Date | Last contact month of year |
| duration | Integer | Last contact duration in seconds |
| campaign | Integer | Number of contacts during this campaign |
| pdays | Integer | Days since last contact from previous campaign (-1 = not contacted) |
| previous | Integer | Number of contacts before this campaign |
| poutcome | Categorical | Outcome of previous marketing campaign |
| y | Binary | **Target:** Has the client subscribed to a term deposit? |

> **Note:** `duration` highly affects the output target and should be discarded for realistic predictive models (included here for benchmarking only).

---

## 2. Data Preprocessing

- **Missing values:** `SimpleImputer` — median for numerical features, most frequent for categorical
- **Normalization:** `StandardScaler` (mean=0, variance=1)
- **Encoding:** `OneHotEncoder` for categorical features to avoid ordinal relationships
- **Dimensionality reduction:** Not applied to features (only 16); PCA used for visualizations only

---

## 3. Exploratory Data Analysis

- Dataset shape: **(45211, 17)**
- **Class imbalance:** `yes` = 11.7%, `no` = 88.3%

![Class Imbalance](images/image6.png)

Most features show weak to negligible correlation (values near 0). Notable exception: `previous` and `pdays` are moderately correlated (0.45).

![Correlation Heatmap](images/image1.png)

The primary demographic represented is middle-aged professionals in blue-collar, management, and technical roles.

![Job Distribution](images/image15.png)

![Age Distribution](images/image17.png)

`balance` and `duration` are heavily right-skewed with significant outliers.

![Balance Distribution](images/image7.png)

![Pdays Distribution](images/image11.png)

Features like `contact` and `poutcome` have class imbalances worth noting for feature selection.

![Contact Distribution](images/image4.png)

![Housing Distribution](images/image5.png)

![Poutcome Distribution](images/image12.png)

---

## 4. Clustering

**Algorithm:** K-Means

Optimal **k = 2** selected via Silhouette Score analysis (score: 0.22), consistent with the binary classification target.

![Silhouette Scores](images/image8.png)

K-Means clustering with PCA (k=2), before outlier removal:

![K-Means Before Outlier Removal](images/image16.png)

After outlier removal, cluster separation is noticeably cleaner and the silhouette score improves to **0.23**:

![K-Means After Outlier Removal](images/image10.png)

Class imbalance limits clustering performance; SMOTE could improve results further.

---

## 5. Outlier Detection

**Algorithm:** Local Outlier Factor (LOF)

Initial parameters (n=20, contamination=0.1) flagged 4,521 outliers — too noisy:

![Outlier Detection - Noisy](images/image14.png)

Final parameters: **contamination = 0.01**, producing **453 outliers** (1% of dataset):

![Outlier Detection - Final](images/image2.png)

Outliers filtered out for all subsequent steps (clustering, classification, feature selection, tuning).

---

## 6. Classification

**Algorithm:** Support Vector Machine (SVM) with class-weight balancing  
**Validation:** 5-fold cross-validation

![ROC Curve - Classification](images/image9.png)

![Confusion Matrix - Classification](images/image18.png)

| Metric | Score |
|---|---|
| Accuracy | 0.9051 |
| Precision | 0.6833 |
| Recall | 0.3517 |
| F1-Score | 0.4642 |
| ROC-AUC | 0.9074 |

| Efficiency | Time |
|---|---|
| Avg. Training Time / Fold | 248.76 seconds |
| Avg. Prediction Time / Fold | 20.77 seconds |

High accuracy and ROC-AUC reflect strong overall performance. Moderate precision reflects a conservative strategy (only targeting high-probability customers). Recall is low — addressed in hyperparameter tuning.

---

## 7. Feature Selection

**Method:** Mutual Information (MI)  
Reduced from 17 features to top **10** to improve efficiency.

| # | Feature | MI Score |
|---|---|---|
| 1 | duration | 0.073354 |
| 2 | pdays | 0.030737 |
| 3 | poutcome_success | 0.029916 |
| 4 | balance | 0.021617 |
| 5 | housing_yes | 0.017223 |
| 6 | poutcome_unknown | 0.016628 |
| 7 | contact_unknown | 0.016561 |
| 8 | previous | 0.015380 |
| 9 | contact_cellular | 0.014241 |
| 10 | housing_no | 0.012566 |

![ROC Curve - Feature Selection](images/image3.png)

![Confusion Matrix - Feature Selection](images/image20.png)

| Metric | Score (change) |
|---|---|
| Accuracy | 0.8993 (-0.6%) |
| Precision | 0.6472 (-6%) |
| Recall | 0.3061 (-14%) |
| F1-Score | 0.4154 (-12%) |
| ROC-AUC | 0.8076 (-11%) |

| Efficiency | Time (change) |
|---|---|
| Avg. Training Time / Fold | 163.74 seconds (-35%) |
| Avg. Prediction Time / Fold | 18.03 seconds (-13%) |

The 35% reduction in training time makes hyperparameter tuning significantly more practical at scale, justifying the tradeoff in model performance.

---

## 8. Hyperparameter Tuning

**Method:** Bayesian Optimization (chosen over Grid/Random Search due to dataset size and deadlock issues with parallelization)

**Optimal parameters found:**
```
kernel = 'rbf'
C = 2.2132
gamma = 'scale'
class_weight = 'balanced'
```

![ROC Curve - Tuned Model](images/image13.png)

![Confusion Matrix - Tuned Model](images/image19.png)

| Metric | Score (change from feature-selected model) |
|---|---|
| Accuracy | 0.8821 |
| Precision | 0.3524 (-46%) |
| Recall | 0.8245 (+169%) |
| F1-Score | 0.4937 (+19%) |
| ROC-AUC | 0.8812 (+10%) |

| Efficiency | Time |
|---|---|
| Avg. Training Time / Fold | 241.44 seconds |
| Avg. Prediction Time / Fold | 32.29 seconds |

Recall more than doubled — critical for a class-imbalanced dataset where missing potential customers is costlier than false positives. This model is the most useful for real-world banking customer acquisition.

---

## 9. Conclusion

Key findings:
- Contact method had a surprisingly significant impact on term deposit subscription probability
- Past campaign outcomes, salary (balance), and occupation were the most important features
- Class imbalance was the central challenge throughout — addressed via class-weighted SVM and careful hyperparameter tuning
- Feature selection delivered a 35% training speedup, enabling practical Bayesian optimization
- Final tuned model achieves **88.1% ROC-AUC** with dramatically improved recall (0.82)

This project demonstrated the importance of preprocessing, outlier handling, and iterative model refinement on real-world imbalanced datasets.

---

## Usage

```bash
python3 main.py
```

**Dependencies:** scikit-learn, pandas, numpy, matplotlib, seaborn
