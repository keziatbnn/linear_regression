import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt

# Load Dataset
train_df = pd.read_csv("training_data.csv")
new_clients = pd.read_csv("new_clients.csv")

# Pisahkan Fitur (X) dan Target (Y)
X = train_df[["Insulation", "Temperature", "Num_Occupants", "Avg_Age", "Home_Size"]]
y = train_df["Heating_Oil"]  # variabel target Y

# Split Data (Training & Testing 80:20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Training Regresi Linear
model = LinearRegression()
model.fit(X_train, y_train)

print("Model berhasil dilatih!")
print("Intercept (a):", model.intercept_)
print("Coefficients (b):", model.coef_)

# Evaluasi Model
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("\n=== Evaluasi Model ===")
print("MAE :", mae)
print("MSE :", mse)
print("RMSE:", rmse)

# Prediksi untuk 42.650 Pelanggan Baru
X_new = new_clients[["Insulation", "Temperature", "Num_Occupants", "Avg_Age", "Home_Size"]]

pred_new = model.predict(X_new)

new_clients["predicted_heating_oil"] = pred_new

# Hasil Prediksi
total_consumption = new_clients["predicted_heating_oil"].sum()

print("\n=== HASIL PREDIKSI ===")
print(f"Total estimasi kebutuhan minyak untuk 42.650 pelanggan baru: {total_consumption:.2f} liter")

# Simpan hasil prediksi ke file
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


# Residual Plot
residuals = y_test - y_pred

plt.figure(figsize=(7,5))
plt.scatter(y_pred, residuals)
plt.axhline(0, color='black', linestyle='--')
plt.xlabel("Predicted Value")
plt.ylabel("Residual")
plt.title("Residual Plot")
plt.grid(True)
plt.show()

