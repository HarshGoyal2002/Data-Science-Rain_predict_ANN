# 🌦️ Rainfall Prediction using Artificial Neural Network (ANN)

## 📌 Project Overview

This project implements an **Artificial Neural Network (ANN)** to predict whether it will rain the next day using the Australian weather dataset. The model analyzes historical weather conditions such as temperature, humidity, pressure, and wind information to classify rainfall occurrence.

The problem is treated as a **binary classification task** where the target variable **RainTomorrow** indicates whether rainfall will occur.

---

## 🎯 Objective

* Build an Artificial Neural Network for rainfall prediction.
* Perform data preprocessing and feature engineering.
* Analyze relationships between weather parameters.
* Evaluate model performance using classification metrics.

---

## 📂 Dataset

Dataset used: **weatherAUS.csv**

The dataset contains daily weather observations across multiple locations in Australia.

---

## 📊 Data Description

The dataset contains various atmospheric measurements recorded daily. Each feature represents weather conditions used for rainfall prediction.

| Feature           | Description                                                                 |
| ----------------- | --------------------------------------------------------------------------- |
| **Location**      | Name of the city in Australia where weather observation was recorded.       |
| **MinTemp**       | Minimum temperature during the day (°C).                                    |
| **MaxTemp**       | Maximum temperature during the day (°C).                                    |
| **Rainfall**      | Amount of rainfall recorded during the day (mm).                            |
| **Evaporation**   | Amount of evaporation during the day (mm).                                  |
| **Sunshine**      | Hours of bright sunshine during the day.                                    |
| **WindGustDir**   | Direction of the strongest wind gust (16 compass directions).               |
| **WindGustSpeed** | Speed of the strongest wind gust (km/h).                                    |
| **WindDir9am**    | Wind direction 10 minutes before 9 AM.                                      |
| **WindDir3pm**    | Wind direction 10 minutes before 3 PM.                                      |
| **WindSpeed9am**  | Wind speed 10 minutes before 9 AM (km/h).                                   |
| **WindSpeed3pm**  | Wind speed 10 minutes before 3 PM (km/h).                                   |
| **Humidity9am**   | Humidity level at 9 AM (%).                                                 |
| **Humidity3pm**   | Humidity level at 3 PM (%).                                                 |
| **Pressure9am**   | Atmospheric pressure at 9 AM (hectopascals).                                |
| **Pressure3pm**   | Atmospheric pressure at 3 PM (hectopascals).                                |
| **Cloud9am**      | Cloud cover at 9 AM (measured in eighths of sky coverage).                  |
| **Cloud3pm**      | Cloud cover at 3 PM (measured in eighths of sky coverage).                  |
| **Temp9am**       | Temperature at 9 AM (°C).                                                   |
| **Temp3pm**       | Temperature at 3 PM (°C).                                                   |
| **RainToday**     | Indicates whether it rained today (Yes/No).                                 |
| **RainTomorrow**  | Target variable indicating whether it will rain tomorrow (1 = Yes, 0 = No). |

**Target Variable:**
👉 **RainTomorrow** — used for rainfall prediction.

---

## 🛠️ Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* TensorFlow / Keras

---

## ⚙️ Project Workflow

### 1️⃣ Data Preprocessing

* Removed columns with high missing values (Evaporation, Sunshine, Cloud9am, Cloud3pm).
* Handled missing values using median (numerical features) and mode (categorical features).
* Converted date column into year, month, and day features.
* Applied label encoding for binary variables.
* Applied one-hot encoding for categorical variables.
* Performed feature scaling using StandardScaler.

---

### 2️⃣ Exploratory Data Analysis

* Target variable distribution analysis.
* Correlation analysis between features.
* Feature relationship visualization.
* Weather pattern analysis.

---

### 3️⃣ Model Development

* Built Artificial Neural Network using TensorFlow/Keras.
* Two hidden layers with ReLU activation.
* Sigmoid activation for binary output.
* Adam optimizer with binary cross-entropy loss.

---

## 🧠 ANN Architecture

* **Input Layer:** Weather features
* **Hidden Layer 1:** 64 neurons (ReLU activation)
* **Hidden Layer 2:** 32 neurons (ReLU activation)
* **Output Layer:** 1 neuron (Sigmoid activation)

---

## 📊 Model Evaluation

* Accuracy score
* Confusion matrix
* Precision, Recall, F1-score
* Training and validation performance analysis

---

## 📈 Results & Insights

* ANN achieved high prediction accuracy (~85%).
* Humidity, pressure, and rainfall were major predictors of rainfall.
* The model successfully captured nonlinear relationships in weather data.
* Data preprocessing significantly improved prediction performance.

---

## 📊 Visualizations

* Target variable distribution
* Correlation analysis
* Humidity vs RainTomorrow
* Pressure vs RainTomorrow
* Model accuracy and loss curves
* Confusion matrix

---

## 🚀 How to Run the Project

### 1. Clone repository

```bash
git clone https://github.com/yourusername/rainfall-ann.git
```

### 2. Install dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow
```

### 3. Run the project

```bash
python ann_model.py
```

---

## 📌 Future Improvements

* Hyperparameter tuning
* Handling class imbalance
* Dropout regularization
* Testing other deep learning architectures

---

## 👤 Author

Harsh Goyal

Artificial Neural Network — Deep Learning Project
