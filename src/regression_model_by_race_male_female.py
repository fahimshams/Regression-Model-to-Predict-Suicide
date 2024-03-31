import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset from Excel sheet
data = pd.read_excel('../data/Processed_Rates_By_Race_Male.xlsx', engine='openpyxl')

# Selecting features and target variable
X = data[['Year']].values  # Feature: Year
y = data['Deaths_per_100k_Resident_Population'].values  # Target: Deaths_per_100k_Resident_Population

# Splitting the data based on different ethnicities
white_data = data[data['Race/Male'] == 'Male: Not Hispanic or Latino: White']
black_data = data[data['Race/Male'] == 'Male: Not Hispanic or Latino: Black or African American']
hispanic_data = data[data['Race/Male'] == 'Male: Hispanic or Latino: All races']
asian_data = data[data['Race/Male'] == 'Male: Not Hispanic or Latino: Asian or Pacific Islander']

# Train-test split for each ethnicity
X_white_train, X_white_test, y_white_train, y_white_test = train_test_split(
    white_data[['Year']].values, white_data['Deaths_per_100k_Resident_Population'].values, test_size=0.2, random_state=42)
X_black_train, X_black_test, y_black_train, y_black_test = train_test_split(
    black_data[['Year']].values, black_data['Deaths_per_100k_Resident_Population'].values, test_size=0.2, random_state=42)
X_hispanic_train, X_hispanic_test, y_hispanic_train, y_hispanic_test = train_test_split(
    hispanic_data[['Year']].values, hispanic_data['Deaths_per_100k_Resident_Population'].values, test_size=0.2, random_state=42)
X_asian_train, X_asian_test, y_asian_train, y_asian_test = train_test_split(
    asian_data[['Year']].values, asian_data['Deaths_per_100k_Resident_Population'].values, test_size=0.2, random_state=42)

# Choose a regression algorithm
model = LinearRegression()

# Train the model for each ethnicity
model.fit(X_white_train, y_white_train)
y_white_pred = model.predict(X_white_test)

model.fit(X_black_train, y_black_train)
y_black_pred = model.predict(X_black_test)

model.fit(X_hispanic_train, y_hispanic_train)
y_hispanic_pred = model.predict(X_hispanic_test)

model.fit(X_asian_train, y_asian_train)
y_asian_pred = model.predict(X_asian_test)

# Plotting the results for each ethnicity
plt.figure(figsize=(10, 6))

plt.scatter(X_white_test, y_white_test, color='blue', label='White')
plt.scatter(X_black_test, y_black_test, color='red', label='Black')
plt.scatter(X_hispanic_test, y_hispanic_test, color='green', label='Hispanic')
plt.scatter(X_asian_test, y_asian_test, color='orange', label='Asian')

# Plotting the regression lines for each ethnicity
plt.plot(X_white_test, y_white_pred, color='blue', linewidth=3)
plt.plot(X_black_test, y_black_pred, color='red', linewidth=3)
plt.plot(X_hispanic_test, y_hispanic_pred, color='green', linewidth=3)
plt.plot(X_asian_test, y_asian_pred, color='orange', linewidth=3)

plt.xlabel('Year')
plt.ylabel('Deaths_per_100k_Resident_Population')
plt.title('Linear Regression Model by Ethnicity')
plt.legend()
plt.show()
