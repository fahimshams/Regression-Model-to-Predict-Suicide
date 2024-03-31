import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
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

# Plotting the data and regression lines for male and female separately
plt.scatter(male_data['Year'], male_data['Deaths_per_100k_Resident_Population'], color='blue', label='Male')
plt.scatter(female_data['Year'], female_data['Deaths_per_100k_Resident_Population'], color='red', label='Female')

plt.plot(male_data['Year'], model_male.predict(male_data[['Year']]), color='blue', linewidth=3, label='Male Regression Line')
plt.plot(female_data['Year'], model_female.predict(female_data[['Year']]), color='red', linewidth=3, label='Female Regression Line')

# Predicting future results (e.g., for years 2019-2025)
future_years = range(2019, 2026)
future_data_male = pd.DataFrame({'Year': future_years})
future_predictions_male = model_male.predict(future_data_male[['Year']])
plt.scatter(future_years, future_predictions_male, color='blue', label='Male Future Predictions')

future_data_female = pd.DataFrame({'Year': future_years})
future_predictions_female = model_female.predict(future_data_female[['Year']])
plt.scatter(future_years, future_predictions_female, color='red', label='Female Future Predictions')

plt.xlabel('Year')
plt.ylabel('Deaths_per_100k_Resident_Population')
plt.title('Linear Regression Model - Male vs Female')
plt.legend()
plt.show()
