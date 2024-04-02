import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load the dataset from Excel sheet
data = pd.read_excel('../data/Processed_Rates_By_Sex.xlsx', engine='openpyxl')

# Splitting the data into male and female subsets
male_data = data[data['Gender'] == 'Male']
female_data = data[data['Gender'] == 'Female']

# Fit linear regression models for male and female data separately
model_male = LinearRegression()
model_male.fit(male_data[['Year']], male_data['Deaths_per_100k_Resident_Population'])

model_female = LinearRegression()
model_female.fit(female_data[['Year']], female_data['Deaths_per_100k_Resident_Population'])

# Predicting future results (e.g., for years 2019-2025) for males
future_years = range(2019, 2026)
future_data_male = pd.DataFrame({'Year': future_years})
future_predictions_male = model_male.predict(future_data_male[['Year']])

# Predicting future results (e.g., for years 2019-2025) for females
future_data_female = pd.DataFrame({'Year': future_years})
future_predictions_female = model_female.predict(future_data_female[['Year']])

# Plotting historical data for males
plt.scatter(male_data['Year'], male_data['Deaths_per_100k_Resident_Population'], color='blue', label='Male Historical Data')

# Plotting historical data for females
plt.scatter(female_data['Year'], female_data['Deaths_per_100k_Resident_Population'], color='red', label='Female Historical Data')

# Plotting the predicted future death rates for males
plt.plot(future_years, future_predictions_male, color='cyan', label='Male Predictions')

# Plotting the predicted future death rates for females
plt.plot(future_years, future_predictions_female, color='magenta', label='Female Predictions')

# Set labels and title
plt.xlabel('Year')
plt.ylabel('Deaths_per_100k_Resident_Population')
plt.title('Linear Regression Prediction Model')

# Show legend
plt.legend()

# Show plot
plt.grid(True)
plt.show()