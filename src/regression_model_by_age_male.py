import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset from Excel sheet
data = pd.read_excel('../data/Processed_Rates_By_Age_Male.xlsx', engine='openpyxl')

# Define age groups
age_groups = ['Male: 10-14 years', 'Male: 15-24 years', 'Male: 25-34 years', 'Male: 35-44 years', 
              'Male: 45-64 years', 'Male: 65 years and over']

# Initialize dictionaries to store coefficients and metrics
coefficients = {}
mse_values = {}
r2_scores = {}

# Plotting the results for each age group
plt.figure(figsize=(10, 6))

# Iterate over each age group
for age_group in age_groups:
    # Filter data for the current age group
    age_data = data[data['Age_Group'] == age_group]
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        age_data[['Year']], age_data['Deaths_per_100k_Resident_Population'], test_size=0.2, random_state=42)
    
    # Fit linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Calculate predictions
    y_pred = model.predict(X_test)
    
    # Store coefficients
    coefficients[age_group] = model.coef_
    
    # Calculate mean squared error
    mse_values[age_group] = mean_squared_error(y_test, y_pred)
    
    # Calculate R-squared score
    r2_scores[age_group] = r2_score(y_test, y_pred)

    # Plot scatter plot and regression line
    plt.scatter(X_test, y_test, label=age_group)
    plt.plot(X_test, y_pred, linewidth=3)

# Print coefficients
print("Coefficients:")
for age_group, coef in coefficients.items():
    print(f"Age Group: {age_group}, Coefficient: {coef}")

# Print mean squared error values
print("\nMean Squared Error:")
for age_group, mse in mse_values.items():
    print(f"Age Group: {age_group}, MSE: {mse}")

# Print R-squared scores
print("\nR-squared Scores:")
for age_group, r2 in r2_scores.items():
    print(f"Age Group: {age_group}, R-squared Score: {r2}")

# Plot settings
plt.xlabel('Year')
plt.ylabel('Deaths_per_100k_Resident_Population')
plt.title('Linear Regression Model by Male Age Group')
plt.legend()
plt.show()