import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt

# Fungsi Regresi Linear
def linear_regression(X, y):
    """
    Menghitung parameter regresi linear:
    beta = (X^T X)^(-1) X^T y
    """
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    beta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
    return beta  

def predict(X, beta):
    """
    Prediksi manual:
    y_hat = beta0 + beta1*x1 + beta2*x2 + ...
    """
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    return X_b.dot(beta)


# Load Dataset
train_df = pd.read_csv("training_data.csv")
new_clients = pd.read_csv("new_clients.csv")

# Pisahkan Fitur (X) dan Target (Y)
X = train_df[["Insulation", "Temperature", "Num_Occupants", "Avg_Age", "Home_Size"]].values
y = train_df["Heating_Oil"].values

# Split Data (Training & Testing 80:20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Training Regresi Linear
beta = linear_regression(X_train, y_train)

print("Model berhasil dilatih secara manual!")
print("Intercept (a):", beta[0])
print("Coefficients (b):", beta[1:])

# Evaluasi Model
y_pred = predict(X_test, beta)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("\n=== Evaluasi Model ===")
print("MAE :", mae)
print("MSE :", mse)
print("RMSE:", rmse)

# Prediksi untuk 42.650 Pelanggan Baru
X_new = new_clients[["Insulation", "Temperature", "Num_Occupants", "Avg_Age", "Home_Size"]].values
pred_new = predict(X_new, beta)
new_clients["predicted_heating_oil"] = pred_new

# Hasil Prediksi
total_consumption = new_clients["predicted_heating_oil"].sum()
print("\n=== HASIL PREDIKSI ===")
print(f"Total estimasi kebutuhan minyak untuk 42.650 pelanggan baru: {total_consumption:.2f} liter")
new_clients.to_csv("prediksi_pelanggan_baru.csv", index=False)
print("Hasil prediksi disimpan ke prediksi_pelanggan_baru.csv")


# Grafik Actual vs Predicted
plt.figure(figsize=(7,5))
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         linestyle='--')  
plt.xlabel("Actual Consumption")
plt.ylabel("Predicted Consumption")
plt.title("Actual vs Predicted (Linear Regression)")
plt.grid(True)
plt.show()


# Grafik Residual Plot
residuals = y_test - y_pred

plt.figure(figsize=(7,5))
plt.scatter(y_pred, residuals)
plt.axhline(0, color='black', linestyle='--')
plt.xlabel("Predicted Value")
plt.ylabel("Residual")
plt.title("Residual Plot")
plt.grid(True)
plt.show()
