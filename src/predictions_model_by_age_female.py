import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load the dataset from Excel sheet
data = pd.read_excel('../data/Processed_Rates_By_Age_Female.xlsx', engine='openpyxl')

# Selecting features and target variable
X = data[['Year']].values  # Feature: Year
y = data['Deaths_per_100k_Resident_Population'].values  # Target: Deaths_per_100k_Resident_Population

# Splitting the data based on different age groups
age_groups = ['Female: 10-14 years', 'Female: 15-24 years', 'Female: 25-34 years', 'Female: 35-44 years', 'Female: 45-64 years', 'Female: 65 years and over']

# Train a linear regression model for each age group
models = {}

for age_group in age_groups:
    age_data = data[data['Age_Group'] == age_group]
    model = LinearRegression().fit(age_data[['Year']], age_data['Deaths_per_100k_Resident_Population'])
    models[age_group] = model

# Predict future results for each age group
future_years = range(2019, 2026)
future_predictions = {}

for age_group, model in models.items():
    future_predictions[age_group] = model.predict(pd.DataFrame({'Year': future_years}))

# Plotting the results for each age group
plt.figure(figsize=(10, 6))

for age_group, color in zip(age_groups, ['blue', 'red', 'green', 'orange', 'purple', 'brown']):
    plt.scatter(data[data['Age_Group'] == age_group]['Year'], 
                data[data['Age_Group'] == age_group]['Deaths_per_100k_Resident_Population'], 
                color=color, label=age_group)
    
    plt.plot(future_years, future_predictions[age_group], color=color, linestyle='--', label=f'Future ({age_group})')

plt.xlabel('Year')
plt.ylabel('Deaths_per_100k_Resident_Population')
plt.title('Linear Regression Model by Female Age Group - Future Predictions')
plt.legend()
plt.show()
