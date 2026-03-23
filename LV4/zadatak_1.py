import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sklearn.linear_model as lm
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score


df = pd.read_csv('data_C02_emission.csv')


# a) Numeric features + split data
numeric_features = [
    'Engine Size (L)',
    'Cylinders',
    'Fuel Consumption City (L/100km)',
    'Fuel Consumption Hwy (L/100km)',
    'Fuel Consumption Comb (L/100km)',
    'Fuel Consumption Comb (mpg)'
]

target_feat = 'CO2 Emissions (g/km)'

if __name__ == '__main__':
    X = df[numeric_features].values
    y = df[target_feat].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=1)

    print(f'Total samples: {df.shape[0]}')
    print(f'Training samples: {X_train.shape[0]}')
    print(f'Test samples: {X_test.shape[0]}')


    # b) Scatter plot C02 vs one numeric feature (Fuel Consumption Comb (L/100km))
    feat_index = numeric_features.index('Fuel Consumption Comb (L/100km)')
    feat_name = numeric_features[feat_index]

    plt.figure()
    plt.scatter(X_train[:, feat_index], y_train, c='blue', s=15, label='Training set')
    plt.scatter(X_test[:, feat_index], y_test, c='red', s=15, label='Test set')
    plt.xlabel(feat_name)
    plt.ylabel(target_feat)
    plt.title(f'{target_feat} vs {feat_name}')
    plt.legend()


    # c) Standardization of input features + histogram
    sc = StandardScaler()
    X_train_n = sc.fit_transform(X_train)
    X_test_n = sc.transform(X_test)


    fig, axes = plt.subplots(1, 2) # 1 row 2 columns
    axes[0].hist(X_train[:, feat_index], bins=30, color='blue', edgecolor='white')
    axes[0].set_xlabel('Value')
    axes[0].set_ylabel('Number of vehicles')
    axes[0].set_title(f'{feat_name} before scaling')

    axes[1].hist(X_train_n[:, feat_index], bins=30, color='red', edgecolor='white')
    axes[1].set_xlabel('Standardized value')
    axes[1].set_ylabel('Number of vehicles')
    axes[1].set_title(f'{feat_name} after scaling')


    # d) Linear regression model
    linearModel = lm.LinearRegression()
    linearModel.fit(X_train_n, y_train)

    print('\n\nLinear regression model parameters')
    print(f'θ0 (intercept) = {linearModel.intercept_:.4f}')
    for i in range(len(numeric_features)):
        print(f'θ{i+1} ({numeric_features[i]}) = {linearModel.coef_[i]:.4f}')

    print('\nModel (equation 4.6)')
    equation_parts = []
    for i in range(len(linearModel.coef_)):
        equation_parts.append(f'{linearModel.coef_[i]:.4f}x{i+1}')
    print(f'ŷ = {linearModel.intercept_:.4f} + {' + '.join(equation_parts)}')


    # e) Prediction + scatter plot: actual vs predicted
    y_test_p = linearModel.predict(X_test_n)

    plt.figure()
    plt.scatter(y_test, y_test_p, c='blue', s=15)
    plt.xlabel('Actual C02 values (g/km)')
    plt.ylabel('Predicted C02 values (g/km)')
    plt.title('Actual vs. predicted values')


    # f) Model evaluation - regression metrics
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


    # g) Changing number of numeric features


    plt.show()