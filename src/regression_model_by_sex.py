import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset from Excel sheet
data = pd.read_excel('../data/Processed_Rates_By_Sex.xlsx', engine='openpyxl')

# Splitting the data into male and female subsets
male_data = data[data['Gender'] == 'Male']
female_data = data[data['Gender'] == 'Female']

# Plotting the data and regression lines for male and female separately
plt.scatter(male_data['Year'], male_data['Deaths_per_100k_Resident_Population'], color='blue', label='Male')
plt.scatter(female_data['Year'], female_data['Deaths_per_100k_Resident_Population'], color='red', label='Female')

# Fit linear regression models for male and female subsets
model_male = LinearRegression()
model_female = LinearRegression()

# Fit the models
model_male.fit(male_data[['Year']], male_data['Deaths_per_100k_Resident_Population'])
model_female.fit(female_data[['Year']], female_data['Deaths_per_100k_Resident_Population'])

# Predictions
y_pred_male = model_male.predict(male_data[['Year']])
y_pred_female = model_female.predict(female_data[['Year']])

# Plotting the regression lines
plt.plot(male_data['Year'], y_pred_male, color='blue', linewidth=3)
plt.plot(female_data['Year'], y_pred_female, color='red', linewidth=3)

plt.xlabel('Year')
plt.ylabel('Deaths_per_100k_Resident_Population')
plt.title('Linear Regression Model - Male vs Female')
plt.legend()
plt.show()

# Calculations
print("Results for Male Linear Regression Model:")
print("Coefficients:", model_male.coef_)
print("Mean Squared Error:", mean_squared_error(male_data['Deaths_per_100k_Resident_Population'], y_pred_male))
print("R-squared Score:", r2_score(male_data['Deaths_per_100k_Resident_Population'], y_pred_male))

print("\nResults for Female Linear Regression Model:")
print("Coefficients:", model_female.coef_)
print("Mean Squared Error:", mean_squared_error(female_data['Deaths_per_100k_Resident_Population'], y_pred_female))
print("R-squared Score:", r2_score(female_data['Deaths_per_100k_Resident_Population'], y_pred_female))