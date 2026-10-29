# EXPERIMENT 8: MODELING AND FORECASTING WITH ARMA PROCESSES
# NAME: MONISHA 
# REG NO: 2305001019

## Aim
To simulate an ARMA time series, fit ARMA models, forecast future values, and compare model performance using AIC, MAE, MSE, and RMSE.

## Objectives
1. To generate an ARMA time series.
2. To plot the series and its ACF and PACF.
3. To split data into training and testing sets.
4. To fit and compare different ARMA models.
5. To forecast future values.
6. To calculate MAE, MSE, and RMSE.
7. To generate an ARMA(2,1) process.

## Software Required
- Python 3
- NumPy
- Pandas
- Matplotlib
- Statsmodels
- Scikit-learn

## Procedure
1. Import the required libraries.
2. Generate ARMA(1,1) time-series data.
3. Plot the time series.
4. Plot the ACF and PACF.
5. Split the data into training and testing sets.
6. Fit an ARMA(1,1) model.
7. Forecast future values.
8. Calculate MAE, MSE, and RMSE.
9. Plot actual and forecast values.
10. Compare different ARMA models using AIC and BIC.
11. Generate an ARMA(2,1) time series.
12. Fit the ARMA(2,1) model and forecast future values.

# PROGRAM 
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Step 1: Generate ARMA(1,1) Data
np.random.seed(42)

ar = np.array([1, -0.7])
ma = np.array([1, 0.4])

data = ArmaProcess(ar, ma).generate_sample(200)
series = pd.Series(data)

# Step 2: Plot Time Series
plt.figure(figsize=(10, 4))
plt.plot(series)
plt.title("ARMA(1,1) Time Series")
plt.grid(True)
plt.show()

# Step 3: Plot ACF and PACF
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
plot_acf(series, lags=20, ax=ax[0])
plot_pacf(series, lags=20, method="ywm", ax=ax[1])
plt.tight_layout()
plt.show()

# Step 4: Split Data
train = series[:160]
test = series[160:]

print("Training Data:", len(train))
print("Testing Data:", len(test))

# Step 5: Fit ARMA(1,1) Model
model = ARIMA(train, order=(1, 0, 1)).fit()
print(model.summary())

# Step 6: Forecast Future Values
forecast = model.forecast(steps=40)

print("Forecast:")
print(forecast)

# Step 7: Calculate Forecast Errors
mae = mean_absolute_error(test, forecast)
mse = mean_squared_error(test, forecast)
rmse = np.sqrt(mse)

print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)

# Step 8: Plot Actual and Forecast Values
plt.figure(figsize=(10, 4))
plt.plot(test.index, test, label="Actual")
plt.plot(test.index, forecast, label="Forecast")
plt.title("Actual vs Forecast")
plt.legend()
plt.grid(True)
plt.show()

# Step 9: Compare Different ARMA Models
orders = [
    (1, 0, 0),
    (0, 0, 1),
    (1, 0, 1),
    (2, 0, 1)
]

results = []

for order in orders:
    fit = ARIMA(train, order=order).fit()

    predicted = fit.forecast(steps=len(test))
    rmse_value = np.sqrt(mean_squared_error(test, predicted))

    results.append((order, fit.aic, fit.bic, rmse_value))

    print("\nARMA Order:", order)
    print("AIC:", fit.aic)
    print("BIC:", fit.bic)
    print("RMSE:", rmse_value)

best_model = min(results, key=lambda item: item[3])
print("\nBest Model Based on RMSE:", best_model[0])

# Step 10: Generate ARMA(2,1) Data
ar2 = np.array([1, -0.5, -0.2])
ma2 = np.array([1, 0.4])

data2 = ArmaProcess(ar2, ma2).generate_sample(200)

plt.figure(figsize=(10, 4))
plt.plot(data2)
plt.title("ARMA(2,1) Time Series")
plt.grid(True)
plt.show()

# Step 11: Fit ARMA(2,1) and Forecast
model2 = ARIMA(data2, order=(2, 0, 1)).fit()
forecast2 = model2.forecast(steps=10)

print("\nARMA(2,1) Forecast:")
print(forecast2)

```



##  Output
<img width="710" height="318" alt="image" src="https://github.com/user-attachments/assets/c6cbe7b3-b67b-4458-ab2a-e8e74d2fdf1f" />
<img width="713" height="323" alt="image" src="https://github.com/user-attachments/assets/46a08862-3ebc-446c-9e6e-7b5477762c2e" />
<img width="913" height="485" alt="image" src="https://github.com/user-attachments/assets/4d687a62-a4f7-404b-a6a2-a5f8d9c7f3f2" />
<img width="833" height="635" alt="image" src="https://github.com/user-attachments/assets/6a3e95f7-6a26-4292-b87d-d4cfad71118a" />
<img width="667" height="416" alt="image" src="https://github.com/user-attachments/assets/b00410f9-a720-4e09-aac3-77f8f3d32fbe" />
<img width="1055" height="462" alt="image" src="https://github.com/user-attachments/assets/ad03bcda-63d1-419f-bba7-3d5976433c8e" />
<img width="377" height="467" alt="image" src="https://github.com/user-attachments/assets/779e9f5b-e72d-4573-b748-3edf531de2ab" />
<img width="1065" height="557" alt="image" src="https://github.com/user-attachments/assets/0803d431-336d-4ad9-9e97-38ead25a5cde" />

## Result
Thus, ARMA processes were generated and analyzed, different ARMA models were fitted and compared using AIC and BIC, and forecasting performance was evaluated using MAE, MSE, and RMSE. The best forecasting model was identified based on the lowest RMSE, and an ARMA(2,1) process was successfully generated and forecasted.
