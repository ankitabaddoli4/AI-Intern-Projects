# Predictive Analytics System Using Machine Learning
# House Price Prediction

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load sample dataset
data = {
    'Area': [800, 900, 1000, 1100, 1200, 1300, 1400, 1500],
    'Bedrooms': [1, 2, 2, 3, 3, 3, 4, 4],
    'Price': [120000, 150000, 170000, 200000, 230000, 260000, 300000, 340000]
}

df = pd.DataFrame(data)

print("Dataset Preview:")
print(df.head())

# Data visualization
sns.pairplot(df)
plt.show()

# Features and target
X = df[['Area', 'Bedrooms']]
y = df['Price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("\nModel Evaluation:")
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Visualization: Actual vs Predicted
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")
plt.show()
