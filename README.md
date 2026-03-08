# 📊 Outlier Detection and Treatment in Placement Dataset

## 📌 Project Overview

This project demonstrates **Outlier Detection and Treatment techniques** using a placement dataset containing:

* **CGPA**
* **Placement Exam Marks**
* **Placement Status**

The goal is to identify and handle **outliers** using statistical techniques such as:

* **Z-Score Method**
* **Trimming**
* **Capping (Winsorization)**

These techniques help improve **data quality and model performance in machine learning**.

---

# 🗂 Dataset

The dataset contains the following columns:

| Column Name          | Description                                   |
| -------------------- | --------------------------------------------- |
| cgpa                 | Student CGPA                                  |
| placement_exam_marks | Placement exam score                          |
| placed               | Placement status (0 = Not placed, 1 = Placed) |

Example dataset preview:

```
cgpa   placement_exam_marks   placed
6.78        12.0                0
7.41         8.0                1
6.81        23.0                0
7.17        11.0                1
```

---

# ⚙️ Technologies Used

* Python 🐍
* Pandas
* NumPy
* Seaborn
* Matplotlib
* Jupyter Notebook

---

# 📊 Data Visualization

Distribution plots were created to understand the distribution of features.

```python
plt.figure(figsize=(16,5))

plt.subplot(1,2,1)
sns.histplot(df['cgpa'], kde=True)

plt.subplot(1,2,2)
sns.histplot(df['placement_exam_marks'], kde=True)

plt.show()
```

This helps identify **skewness and potential outliers** in the dataset.

---

# 📉 Detecting Outliers

### Step 1: Calculate Statistics

```python
print("Mean value of cgpa", df['cgpa'].mean())
print("Std value of cgpa", df['cgpa'].std())
print("Min value of cgpa", df['cgpa'].min())
print("Max value of cgpa", df['cgpa'].max())
```

---

# 📏 Boundary Calculation (3 Standard Deviation Rule)

```python
upper_limit = df['cgpa'].mean() + 3 * df['cgpa'].std()
lower_limit = df['cgpa'].mean() - 3 * df['cgpa'].std()
```

---

# ✂️ Trimming Method

Trimming removes extreme values outside the acceptable range.

```python
new_df = df[(df['cgpa'] < upper_limit) & (df['cgpa'] > lower_limit)]
```

Result:

```
995 rows × 4 columns
```

Outliers are removed from the dataset.

---

# 📐 Z-Score Method

Z-score measures how far a value is from the mean.

Formula:

```
Z = (X − Mean) / Standard Deviation
```

Implementation:

```python
df['cgpa_zscore'] = (df['cgpa'] - df['cgpa'].mean()) / df['cgpa'].std()
```

Detecting Outliers:

```python
df[(df['cgpa_zscore'] > 3) | (df['cgpa_zscore'] < -3)]
```

---

# 🧢 Capping Method (Winsorization)

Instead of removing outliers, we **cap them within limits**.

```python
df['cgpa'] = np.where(
    df['cgpa'] > upper_limit,
    upper_limit,
    np.where(
        df['cgpa'] < lower_limit,
        lower_limit,
        df['cgpa']
    )
)
```

This ensures the dataset **retains all rows while controlling extreme values**.

---

# 📈 Final Dataset Shape

```
(1000, 4)
```

After capping, all values remain within the acceptable range.

---

# 🚀 Key Learnings

* Understanding **data distribution**
* Detecting **outliers using statistical techniques**
* Applying **Trimming and Capping**
* Using **Z-score for anomaly detection**
* Improving dataset quality for **machine learning models**

---

# 📌 Future Improvements

* Apply **IQR Outlier Detection**
* Train **Machine Learning Models**
* Compare model performance **before and after outlier treatment**

---

# 👨‍💻 Author

**Aryan Nigam**

Power BI & Data Analytics Enthusiast
Learning **Machine Learning & Data Science**

---

⭐ If you found this project helpful, please consider **starring the repository**.
