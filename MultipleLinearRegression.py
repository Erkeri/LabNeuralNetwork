import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import category_encoders as ce
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

dataset = pd.read_csv('D:\\car_data.csv', encoding = 'ISO-8859-1')

encoder = ce.BinaryEncoder(cols=['Fuel_Type', 'Selling_type', 'Transmission']).fit(dataset)
numeric_dataset = encoder.transform(dataset)

X = numeric_dataset[['Year', 'Selling_Price', 'Driven_kms', 'Fuel_Type_0', 'Selling_type_0', 'Transmission_0']]
y = numeric_dataset['Present_Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(df)

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))