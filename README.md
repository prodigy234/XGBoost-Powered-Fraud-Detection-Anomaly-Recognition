
# 🛡️ Fraud Detection System using XGBoost and Streamlit

This project is a synthetic fraud detection system built using Python, Streamlit for the UI, and XGBoost for the machine learning model. It simulates transaction data and predicts the likelihood of a transaction being fraudulent.

---

This highly smart fraud detector which I developed can be accessed live on streamlit [Here](https://smart-fraud-detector.streamlit.app/)

---

## 📬 Author

**Gbenga Kajola**

[LinkedIn](https://www.linkedin.com/in/kajolagbenga)

[Certified_Data_Scientist](https://www.datacamp.com/certificate/DSA0012312825030)

[Certified_Data_Analyst](https://www.datacamp.com/certificate/DAA0018583322187)

[Certified_SQL_Database_Programmer](https://www.datacamp.com/certificate/SQA0019722049554)

---


## 🚀 Features

- Generates a synthetic dataset of transactions with labeled fraud and legitimate cases.
- Uses `XGBoostClassifier` to train a fraud detection model.
- Scales features using `StandardScaler`.
- Provides a simple and interactive web UI using Streamlit.
- Allows real-time prediction of transactions to detect fraud.
- Displays dataset statistics and sample data.

---

## 🧠 Machine Learning Details

- **Data**: Synthetic, generated with specific statistical parameters.
- **Features**:
  - `TransactionAmount`
  - `TransactionTime`
  - `TransactionHour`
  - `UserRiskScore`
- **Target**: `IsFraud` (0 = Legitimate, 1 = Fraudulent)
- **Model**: XGBoost Classifier
- **Preprocessing**: StandardScaler normalization

---

## 🛠️ Installation

```bash
git clone https://github.com/prodigy234/XGBoost-Powered-Fraud-Detection-Anomaly-Recognition.git
cd XGBoost-Powered Fraud Detection Anomaly Recognition
pip install -r requirements.txt
```

---

## ▶️ How to Run

```bash
streamlit run app.py
```

---

## 📂 Project Structure

```
📁 XGBoost-Powered Fraud Detection Anomaly Recognition/
├── fraud_detection.py                 # Main Streamlit application file
├── my_image.jpg                       # My image
├── scaler.pkl                         # Saved StandardScaler object
├── xgb_model.pkl                      # Trained XGBoost model
├── README.md                          # Project documentation
└── requirements.txt                   # Required Python packages
```

---

## 💻 Streamlit Interface

- Accepts user inputs for:
  - Transaction Amount
  - Transaction Time
  - Transaction Hour (0-23)
  - User Risk Score (0.0 - 1.0)
- Displays prediction result:
  - ✅ Legitimate Transaction
  - 🛑 Fraudulent Transaction Detected!
- Option to show data summary and sample

---

## 📊 Sample Output

![](https://via.placeholder.com/800x400.png?text=Streamlit+App+Demo)

---

## 📦 Dependencies

```
numpy
pandas
streamlit
joblib
scikit-learn
xgboost
```

Install using:

```bash
pip install -r requirements.txt
```

---

## 📈 Model Evaluation

The model is trained and evaluated using:
- `classification_report`
- `roc_auc_score`

Due to the synthetic and imbalanced nature of the data (1% fraud), evaluation metrics should be interpreted accordingly.

---

## 📄 License

This project is licensed under the MIT License.
