import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import sklearn.linear_model as lm
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score

from zadatak_1 import df, numeric_features, target_feat

ohe = OneHotEncoder()
X_fuel = ohe.fit_transform(df[['Fuel Type']]).toarray()

X = np.hstack((df[numeric_features].values, X_fuel))
y = df[target_feat].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=1)

linearModel = lm.LinearRegression()
linearModel.fit(X_train, y_train)

y_test_p = linearModel.predict(X_test)

plt.figure()
plt.scatter(y_test, y_test_p, c='blue', s=15)
plt.xlabel('Actual C02 values (g/km)')
plt.ylabel('Predicted C02 values (g/km)')
plt.title('Actual vs. predicted values')

mse = mean_squared_error(y_test, y_test_p)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_test_p)
mape = mean_absolute_percentage_error(y_test, y_test_p) * 100
r2 = r2_score(y_test, y_test_p)

print('\n\nRegression metrics')
print(f'MSE = {mse:.4f}')
print(f'RMSE = {rmse:.4f}')
print(f'MAE = {mae:.4f}')
print(f'MAPE = {mape:.4f}%')
print(f'R2 = {r2:.4f}')


errors = np.abs(y_test - y_test_p)
max_error_index = np.argmax(errors)

_, df_test = train_test_split(df, test_size=0.2, random_state=1)
worst_vehicle = df_test.iloc[max_error_index]

print(f'\n\nMaximum prediction error: {errors[max_error_index]:.2f} g/km')
print(f'Vehicle: {worst_vehicle['Make']} {worst_vehicle['Model']}')


plt.show()