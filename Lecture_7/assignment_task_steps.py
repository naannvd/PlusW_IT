#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("owid-energy-data.csv")

# Check dataset info
print(df.info())

# Display first few rows
print(df.head())

# Check for missing values
print(df.isnull().sum())

# Visualize energy consumption trends
plt.figure(figsize=(12, 6))
sns.histplot(df['primary_energy_consumption'], bins=30, kde=True)
plt.title("Distribution of Primary Energy Consumption")
plt.show()
#%%
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Custom Linear Regression using Gradient Descent
class LinearRegressionGD:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.theta = None

    def fit(self, X, y):
        X = np.c_[np.ones(X.shape[0]), X]  # Add bias term
        self.theta = np.zeros(X.shape[1])  # Initialize weights

        for _ in range(self.epochs):
            gradients = -2 / X.shape[0] * X.T @ (y - X @ self.theta)
            self.theta -= self.learning_rate * gradients

    def predict(self, X):
        if self.theta is None:
            raise ValueError("Model has not been trained. Call fit() first.")
        X = np.c_[np.ones(X.shape[0]), X]  # Add bias term
        return X @ self.theta

# Load and prepare your dataset here
# For demonstration, let's simulate some clean energy data
# Replace this with your real dataset (e.g., owid-energy-data.csv)
df = pd.read_csv("owid-energy-data.csv")
df.dropna(inplace=True)

# Simulate temperature column if it doesn't exist
if 'Temperature' not in df.columns:
    df['Temperature'] = np.random.randint(0, 40, size=len(df))

# Choose your target column
target_col = [col for col in df.columns if "consumption" in col.lower()]
if not target_col:
    raise ValueError("No energy consumption column found.")
target_col = target_col[0]

# Select numeric features only
if 'Year' not in df.columns:
    if 'year' in df.columns:
        df['Year'] = pd.to_datetime(df['year'], errors='coerce').dt.year
    else:
        df['Year'] = np.random.randint(2000, 2025, size=len(df))

if 'Temperature' not in df.columns:
    df['Temperature'] = np.random.randint(0, 40, size=len(df))

# Now this will work
features = ['Year', 'Temperature']
X = df[features]
y = df[target_col]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# Train using gradient descent
model_gd = LinearRegressionGD(learning_rate=0.001, epochs=10000)
model_gd.fit(X_train_scaled, y.to_numpy())

# Predict
y_pred_gd = model_gd.predict(X_test_scaled)

# Clean NaNs if present
y_test_clean = y_test[~y_test.isnull()]
y_pred_clean = y_pred_gd[~y_test.isnull()]

# Evaluate
mse_gd = mean_squared_error(y_test_clean, y_pred_clean)
r2_gd = r2_score(y_test_clean, y_pred_clean)

print(f"✅ Gradient Descent Model Evaluation:")
print(f"   MSE: {mse_gd:.2f}")
print(f"   R²: {r2_gd:.2f}")


#%%
import streamlit as st
import numpy as np

# Ensure 'Year' column exists
if 'Year' not in df.columns:
    if 'year' in df.columns:
        df['Year'] = pd.to_datetime(df['year'], errors='coerce').dt.year
    else:
        df['Year'] = np.random.randint(2000, 2025, size=len(df))  # fallback dummy year

# Streamlit UI
st.title("Energy Consumption Prediction")
st.write(f"MSE: {mse_gd:.2f}, R²: {r2_gd:.2f}")

# User input for prediction
st.sidebar.header('Predict Future Energy Consumption')

year_min = int(df['Year'].min())
year_max = int(df['Year'].max())

if year_min == year_max:
    year = st.sidebar.number_input("Enter Year", value=year_min)
else:
    year = st.sidebar.slider("Select Year", year_min, year_max, step=1)

temperature = st.sidebar.slider("Select Temperature", 0, 40, step=1)

# Make prediction
input_data = np.array([[year, temperature]])
input_data_scaled = scaler.transform(input_data)
prediction = model_gd.predict(input_data_scaled)[0]

st.sidebar.write(f'Predicted Energy Consumption: {prediction:.2f}')




