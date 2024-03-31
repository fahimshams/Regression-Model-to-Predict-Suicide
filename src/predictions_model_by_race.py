import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

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

# Train a linear regression model for each ethnicity
model_white = LinearRegression().fit(white_data[['Year']], white_data['Deaths_per_100k_Resident_Population'])
model_black = LinearRegression().fit(black_data[['Year']], black_data['Deaths_per_100k_Resident_Population'])
model_hispanic = LinearRegression().fit(hispanic_data[['Year']], hispanic_data['Deaths_per_100k_Resident_Population'])
model_asian = LinearRegression().fit(asian_data[['Year']], asian_data['Deaths_per_100k_Resident_Population'])

# Predict future results for each ethnicity
future_years = range(2019, 2026)

future_white_predictions = model_white.predict(pd.DataFrame({'Year': future_years}))
future_black_predictions = model_black.predict(pd.DataFrame({'Year': future_years}))
future_hispanic_predictions = model_hispanic.predict(pd.DataFrame({'Year': future_years}))
future_asian_predictions = model_asian.predict(pd.DataFrame({'Year': future_years}))

# Plotting the results for each ethnicity
plt.figure(figsize=(10, 6))

plt.scatter(white_data['Year'], white_data['Deaths_per_100k_Resident_Population'], color='blue', label='White')
plt.scatter(black_data['Year'], black_data['Deaths_per_100k_Resident_Population'], color='red', label='Black')
plt.scatter(hispanic_data['Year'], hispanic_data['Deaths_per_100k_Resident_Population'], color='green', label='Hispanic')
plt.scatter(asian_data['Year'], asian_data['Deaths_per_100k_Resident_Population'], color='orange', label='Asian')

# Plotting the future predictions for each ethnicity
plt.plot(future_years, future_white_predictions, color='blue', linestyle='--', label='Future (White)')
plt.plot(future_years, future_black_predictions, color='red', linestyle='--', label='Future (Black)')
plt.plot(future_years, future_hispanic_predictions, color='green', linestyle='--', label='Future (Hispanic)')
plt.plot(future_years, future_asian_predictions, color='orange', linestyle='--', label='Future (Asian)')

plt.xlabel('Year')
plt.ylabel('Deaths_per_100k_Resident_Population')
plt.title('Linear Regression Model by Ethnicity - Future Predictions')
plt.legend()
plt.show()
