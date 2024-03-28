#By Sex
#1. Load the processed dataset.
#2. Define features (Year, Gender) and the target variable (Deaths_per_100k_Resident_Population).
#3. Convert categorical variables into dummy/indicator variables.
#4. Split the data into training and testing sets.
#5. Initialize and train a linear regression model.
#6. Predict on the testing set and evaluate the model using mean squared error.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the processed dataset
data = pd.read_excel('../data/Processed_Suicide_Rates.xlsx', sheet_name='By sex')

# Define features and target variable
X = data[['Year', 'Gender']]
y = data['Deaths_per_100k_Resident_Population']

# Convert categorical variables into dummy/indicator variables
X = pd.get_dummies(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Predict on the testing set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Save the trained model for later use:
# import joblib
# joblib.dump(model, 'linear_regression_model.pkl')
