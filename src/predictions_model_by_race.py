import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load the dataset from Excel sheet for male
male_data = pd.read_excel('../data/Processed_Rates_By_Race_Male.xlsx', engine='openpyxl')

# Load the dataset from Excel sheet for female
female_data = pd.read_excel('../data/Processed_Rates_By_Race_Female.xlsx', engine='openpyxl')

# Selecting features and target variable for male
X_male = male_data[['Year']].values
y_male = male_data['Deaths_per_100k_Resident_Population'].values

# Selecting features and target variable for female
X_female = female_data[['Year']].values
y_female = female_data['Deaths_per_100k_Resident_Population'].values

# Splitting the data based on different ethnicities for male
white_data_male = male_data[male_data['Race/Male'] == 'Male: Not Hispanic or Latino: White']
black_data_male = male_data[male_data['Race/Male'] == 'Male: Not Hispanic or Latino: Black or African American']
hispanic_data_male = male_data[male_data['Race/Male'] == 'Male: Hispanic or Latino: All races']
asian_data_male = male_data[male_data['Race/Male'] == 'Male: Not Hispanic or Latino: Asian or Pacific Islander']

# Splitting the data based on different ethnicities for female
white_data_female = female_data[female_data['Race/Female'] == 'Female: Not Hispanic or Latino: White']
black_data_female = female_data[female_data['Race/Female'] == 'Female: Not Hispanic or Latino: Black or African American']
hispanic_data_female = female_data[female_data['Race/Female'] == 'Female: Hispanic or Latino: All races']
asian_data_female = female_data[female_data['Race/Female'] == 'Female: Not Hispanic or Latino: Asian or Pacific Islander']

# Train a linear regression model for each ethnicity for male
model_white_male = LinearRegression().fit(white_data_male[['Year']], white_data_male['Deaths_per_100k_Resident_Population'])
model_black_male = LinearRegression().fit(black_data_male[['Year']], black_data_male['Deaths_per_100k_Resident_Population'])
model_hispanic_male = LinearRegression().fit(hispanic_data_male[['Year']], hispanic_data_male['Deaths_per_100k_Resident_Population'])
model_asian_male = LinearRegression().fit(asian_data_male[['Year']], asian_data_male['Deaths_per_100k_Resident_Population'])

# Train a linear regression model for each ethnicity for female
model_white_female = LinearRegression().fit(white_data_female[['Year']], white_data_female['Deaths_per_100k_Resident_Population'])
model_black_female = LinearRegression().fit(black_data_female[['Year']], black_data_female['Deaths_per_100k_Resident_Population'])
model_hispanic_female = LinearRegression().fit(hispanic_data_female[['Year']], hispanic_data_female['Deaths_per_100k_Resident_Population'])
model_asian_female = LinearRegression().fit(asian_data_female[['Year']], asian_data_female['Deaths_per_100k_Resident_Population'])

# Predict future results for each ethnicity for male
future_years = range(2019, 2026)

future_white_predictions_male = model_white_male.predict(pd.DataFrame({'Year': future_years}))
future_black_predictions_male = model_black_male.predict(pd.DataFrame({'Year': future_years}))
future_hispanic_predictions_male = model_hispanic_male.predict(pd.DataFrame({'Year': future_years}))
future_asian_predictions_male = model_asian_male.predict(pd.DataFrame({'Year': future_years}))

# Predict future results for each ethnicity for female
future_white_predictions_female = model_white_female.predict(pd.DataFrame({'Year': future_years}))
future_black_predictions_female = model_black_female.predict(pd.DataFrame({'Year': future_years}))
future_hispanic_predictions_female = model_hispanic_female.predict(pd.DataFrame({'Year': future_years}))
future_asian_predictions_female = model_asian_female.predict(pd.DataFrame({'Year': future_years}))

# Plotting the results for each ethnicity for male
plt.figure(figsize=(10, 6))

plt.scatter(white_data_male['Year'], white_data_male['Deaths_per_100k_Resident_Population'], color='blue', label='White (Male)')
plt.scatter(black_data_male['Year'], black_data_male['Deaths_per_100k_Resident_Population'], color='red', label='Black (Male)')
plt.scatter(hispanic_data_male['Year'], hispanic_data_male['Deaths_per_100k_Resident_Population'], color='green', label='Hispanic (Male)')
plt.scatter(asian_data_male['Year'], asian_data_male['Deaths_per_100k_Resident_Population'], color='orange', label='Asian (Male)')

# Plotting the future predictions for each ethnicity for male
plt.plot(future_years, future_white_predictions_male, color='blue', linestyle='--', label='Future (White) (Male)')
plt.plot(future_years, future_black_predictions_male, color='red', linestyle='--', label='Future (Black) (Male)')
plt.plot(future_years, future_hispanic_predictions_male, color='green', linestyle='--', label='Future (Hispanic) (Male)')
plt.plot(future_years, future_asian_predictions_male, color='orange', linestyle='--', label='Future (Asian) (Male)')

# Plotting the results for each ethnicity for female
plt.scatter(white_data_female['Year'], white_data_female['Deaths_per_100k_Resident_Population'], color='blue', marker='x', label='White (Female)')
plt.scatter(black_data_female['Year'], black_data_female['Deaths_per_100k_Resident_Population'], color='red', marker='x', label='Black (Female)')
plt.scatter(hispanic_data_female['Year'], hispanic_data_female['Deaths_per_100k_Resident_Population'], color='green', marker='x', label='Hispanic (Female)')
plt.scatter(asian_data_female['Year'], asian_data_female['Deaths_per_100k_Resident_Population'], color='orange', marker='x', label='Asian (Female)')

# Plotting the future predictions for each ethnicity for female
plt.plot(future_years, future_white_predictions_female, color='blue', linestyle=':', label='Future (White) (Female)')
plt.plot(future_years, future_black_predictions_female, color='red', linestyle=':', label='Future (Black) (Female)')
plt.plot(future_years, future_hispanic_predictions_female, color='green', linestyle=':', label='Future (Hispanic) (Female)')
plt.plot(future_years, future_asian_predictions_female, color='orange', linestyle=':', label='Future (Asian) (Female)')

plt.xlabel('Year')
plt.ylabel('Deaths_per_100k_Resident_Population')
plt.title('Linear Regression Model by Ethnicity - Future Predictions (Male and Female)')
plt.legend()
plt.show()
