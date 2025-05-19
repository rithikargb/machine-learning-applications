import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing

data = fetch_california_housing()
housing_df = pd.DataFrame(data.data, columns=data.feature_names)
housing_df['Price'] = data.target

features = ['AveRooms', 'AveOccup']
X = housing_df[features]
y = housing_df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

plt.scatter(X_test['AveRooms'], y_test, color='blue', label='Actual Prices')
plt.scatter(X_test['AveRooms'], y_pred, color='red', label='Predicted Prices')
plt.title('Regression Line on Scatter Plot')
plt.xlabel('AveRooms')
plt.ylabel('Price')
plt.legend()
plt.show()
