import numpy as np
import pandas as pd
import streamlit as st
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score

# Step 1: Generate Synthetic Fraud Dataset with More Features
def generate_synthetic_data(n_samples=10000, fraud_ratio=0.01):
    np.random.seed(42)
    
    n_fraud = int(n_samples * fraud_ratio)
    n_legit = n_samples - n_fraud
    
    # Legitimate transactions
    legit = np.random.normal(loc=[50, 1], scale=[15, 0.5], size=(n_legit, 2))
    legit_labels = np.zeros(n_legit)
    
    # Fraudulent transactions with added noise
    fraud = np.random.normal(loc=[200, 5], scale=[50, 2], size=(n_fraud, 2))
    fraud[:, 0] += np.random.normal(0, 20, n_fraud)  # Add noise to TransactionAmount
    fraud[:, 1] += np.random.normal(0, 1, n_fraud)  # Add noise to TransactionTime
    fraud_labels = np.ones(n_fraud)
    
    data = np.vstack((legit, fraud))
    labels = np.concatenate((legit_labels, fraud_labels))
    
    df = pd.DataFrame(data, columns=['TransactionAmount', 'TransactionTime'])
    df['TransactionHour'] = np.random.randint(0, 24, df.shape[0])  # Random transaction hour
    df['UserRiskScore'] = np.random.uniform(0, 1, df.shape[0])  # Simulated user risk score
    df['IsFraud'] = labels
    
    return df

# Generate data
data = generate_synthetic_data()

# Step 2: Preprocessing & Splitting
y = data['IsFraud']
X = data.drop(columns=['IsFraud'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 3: Train Machine Learning Model (XGBoost) with Tuned Hyperparameters
xgb = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, eval_metric='logloss')
xgb.fit(X_train_scaled, y_train)

# Save Model and Scaler
joblib.dump(scaler, "scaler.pkl")
joblib.dump(xgb, "xgb_model.pkl")

# Step 4: Streamlit UI
def main():
    st.title("Fraud Detection System")
    st.sidebar.header("User Input")
    
    # User input fields
    amount = st.sidebar.number_input("Transaction Amount", min_value=0.0, step=0.1)
    time = st.sidebar.number_input("Transaction Time", min_value=0.0, step=0.1)
    hour = st.sidebar.slider("Transaction Hour", 0, 23, 12)
    risk = st.sidebar.slider("User Risk Score", 0.0, 1.0, 0.5)
    
    if st.sidebar.button("Predict Fraud"):
        model = joblib.load("xgb_model.pkl")
        scaler = joblib.load("scaler.pkl")
        
        input_data = pd.DataFrame([[amount, time, hour, risk]],
                                  columns=['TransactionAmount', 'TransactionTime', 'TransactionHour', 'UserRiskScore'])
        prediction = model.predict(scaler.transform(input_data))
        
        if prediction[0] == 1:
            st.error("🛑 Fraudulent Transaction Detected!")
        else:
            st.success("✅ Legitimate Transaction")
    
    # Display dataset statistics
    if st.checkbox("Show Dataset Summary"):
        st.write(data.describe())
    
    if st.checkbox("Show Sample Data"):
        st.write(data.head())

    # Footer
    st.markdown("---")
    st.markdown("# About the Developer")
    # Display developer image
    st.image("my_image.jpg", width=150)
    st.markdown("## **Kajola Gbenga**")

    st.markdown(
        """
    📇 Certified Data Analyst | Certified Data Scientist | Certified SQL Programmer | Mobile App Developer | AI/ML Engineer

    🔗 [LinkedIn](https://www.linkedin.com/in/kajolagbenga)  
    📜 [View My Certifications & Licences](https://www.datacamp.com/portfolio/kgbenga234)  
    💻 [GitHub](https://github.com/prodigy234)  
    🌐 [Portfolio](https://kajolagbenga.netlify.app/)  
    📧 k.gbenga234@gmail.com
    """
    )


if __name__ == "__main__":
    main()
