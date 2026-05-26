# Credit Card Fraud Detection: Machine Learning vs. Deep Learning

An end-to-end data science project benchmarking classical Machine Learning models against an Artificial Neural Network (ANN) to detect fraudulent credit card transactions. 

The dataset used is highly imbalanced, containing only **0.17%** fraud cases. This project demonstrates advanced techniques for handling class imbalance, including **StratifiedShuffleSplit** and **SMOTE (Synthetic Minority Over-sampling Technique)**.

## 🚀 Project Overview & Workflow
1. **Exploratory Data Analysis (EDA):** Analyzed feature distributions and verified severe class imbalance.
2. **Data Splitting:** Applied `StratifiedShuffleSplit` to ensure training and testing splits mirror the real-world fraud ratio.
3. **Data Balancing:** Implemented `SMOTE` *only* on the training data to create synthetic fraud profiles, preventing the "Accuracy Paradox."
4. **Model Benchmarking:** Trained and compared **Logistic Regression**, **Random Forest**, and a **Deep Learning Neural Network (ANN)**.

---

## 📊 Model Performance Comparison

Because fraud detection requires minimizing financial loss (high Recall) while maintaining user experience (high Precision), models were evaluated using a Classification Report.

| Model | Precision (Class 1) | Recall (Class 1) | F1-Score (Class 1) | Performance Status |
| :--- | :---: | :---: | :---: | :--- |
| **1. Logistic Regression** | 0.05 | **0.87** | 0.10 | Highly paranoid; flags too many honest customers. |
| **2. Random Forest** | **0.91** | 0.77 | **0.83** | **Project Winner.** Safest balance for commercial banking. |
| **3. Deep Learning (ANN)** | 0.68 | 0.80 | 0.74 | Reliable alternative with strong balanced metrics. |

### Key Takeaway:
While the **Deep Learning ANN** caught more fraud cases (80% Recall), the **Random Forest** achieved the highest overall capability with an **F1-Score of 0.83**. It stopped the massive "false alarm" paranoia flaw observed in Logistic Regression while keeping genuine customer friction at an absolute minimum (91% Precision).

---

## 🛠️ Tech Stack & Libraries
* **Language:** Python
* **Machine Learning:** Scikit-Learn, Imbalanced-Learn (SMOTE)
* **Deep Learning:** TensorFlow / Keras
* **Visualization:** Matplotlib, Seaborn, Pandas, NumPy

---

## 💻 How to Run This Project

1. Clone the repository:
   ```bash
   git clone [https://github.com/YOUR_USERNAME/Credit-Card-Fraud-Detection.git](https://github.com/YOUR_USERNAME/Credit-Card-Fraud-Detection.git)