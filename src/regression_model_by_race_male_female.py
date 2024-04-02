import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the datasets from Excel sheets
male_data = pd.read_excel('../data/Processed_Rates_By_Race_Male.xlsx', engine='openpyxl')
female_data = pd.read_excel('../data/Processed_Rates_By_Race_Female.xlsx', engine='openpyxl')

# Concatenate male and female data
data = pd.concat([male_data, female_data])

# Selecting features and target variable
X = data[['Year']].values  # Feature: Year
y = data['Deaths_per_100k_Resident_Population'].values  # Target: Deaths_per_100k_Resident_Population

# Splitting the data based on different ethnicities and genders
white_male_data = male_data[male_data['Race/Male'] == 'Male: Not Hispanic or Latino: White']
black_male_data = male_data[male_data['Race/Male'] == 'Male: Not Hispanic or Latino: Black or African American']
hispanic_male_data = male_data[male_data['Race/Male'] == 'Male: Hispanic or Latino: All races']
asian_male_data = male_data[male_data['Race/Male'] == 'Male: Not Hispanic or Latino: Asian or Pacific Islander']

white_female_data = female_data[female_data['Race/Female'] == 'Female: Not Hispanic or Latino: White']
black_female_data = female_data[female_data['Race/Female'] == 'Female: Not Hispanic or Latino: Black or African American']
hispanic_female_data = female_data[female_data['Race/Female'] == 'Female: Hispanic or Latino: All races']
asian_female_data = female_data[female_data['Race/Female'] == 'Female: Not Hispanic or Latino: Asian or Pacific Islander']

# Train-test split for each ethnicity and gender
X_white_male_train, X_white_male_test, y_white_male_train, y_white_male_test = train_test_split(
    white_male_data[['Year']].values, white_male_data['Deaths_per_100k_Resident_Population'].values, test_size=0.2, random_state=42)
X_black_male_train, X_black_male_test, y_black_male_train, y_black_male_test = train_test_split(
    black_male_data[['Year']].values, black_male_data['Deaths_per_100k_Resident_Population'].values, test_size=0.2, random_state=42)
X_hispanic_male_train, X_hispanic_male_test, y_hispanic_male_train, y_hispanic_male_test = train_test_split(
    hispanic_male_data[['Year']].values, hispanic_male_data['Deaths_per_100k_Resident_Population'].values, test_size=0.2, random_state=42)
X_asian_male_train, X_asian_male_test, y_asian_male_train, y_asian_male_test = train_test_split(
    asian_male_data[['Year']].values, asian_male_data['Deaths_per_100k_Resident_Population'].values, test_size=0.2, random_state=42)

X_white_female_train, X_white_female_test, y_white_female_train, y_white_female_test = train_test_split(
    white_female_data[['Year']].values, white_female_data['Deaths_per_100k_Resident_Population'].values, test_size=0.2, random_state=42)
X_black_female_train, X_black_female_test, y_black_female_train, y_black_female_test = train_test_split(
    black_female_data[['Year']].values, black_female_data['Deaths_per_100k_Resident_Population'].values, test_size=0.2, random_state=42)
X_hispanic_female_train, X_hispanic_female_test, y_hispanic_female_train, y_hispanic_female_test = train_test_split(
    hispanic_female_data[['Year']].values, hispanic_female_data['Deaths_per_100k_Resident_Population'].values, test_size=0.2, random_state=42)
X_asian_female_train, X_asian_female_test, y_asian_female_train, y_asian_female_test = train_test_split(
    asian_female_data[['Year']].values, asian_female_data['Deaths_per_100k_Resident_Population'].values, test_size=0.2, random_state=42)

# Choose a regression algorithm
model = LinearRegression()

# Train the model for each ethnicity and gender
model.fit(X_white_male_train, y_white_male_train)
y_white_male_pred = model.predict(X_white_male_test)

model.fit(X_black_male_train, y_black_male_train)
y_black_male_pred = model.predict(X_black_male_test)

model.fit(X_hispanic_male_train, y_hispanic_male_train)
y_hispanic_male_pred = model.predict(X_hispanic_male_test)

model.fit(X_asian_male_train, y_asian_male_train)
y_asian_male_pred = model.predict(X_asian_male_test)

model.fit(X_white_female_train, y_white_female_train)
y_white_female_pred = model.predict(X_white_female_test)

model.fit(X_black_female_train, y_black_female_train)
y_black_female_pred = model.predict(X_black_female_test)

model.fit(X_hispanic_female_train, y_hispanic_female_train)
y_hispanic_female_pred = model.predict(X_hispanic_female_test)

model.fit(X_asian_female_train, y_asian_female_train)
y_asian_female_pred = model.predict(X_asian_female_test)

# Plotting the results for each ethnicity and gender
plt.figure(figsize=(10, 6))

plt.scatter(X_white_male_test, y_white_male_test, color='blue', label='White Male')
plt.scatter(X_black_male_test, y_black_male_test, color='red', label='Black Male')
plt.scatter(X_hispanic_male_test, y_hispanic_male_test, color='green', label='Hispanic Male')
plt.scatter(X_asian_male_test, y_asian_male_test, color='orange', label='Asian Male')

plt.scatter(X_white_female_test, y_white_female_test, color='lightblue', label='White Female')
plt.scatter(X_black_female_test, y_black_female_test, color='pink', label='Black Female')
plt.scatter(X_hispanic_female_test, y_hispanic_female_test, color='lightgreen', label='Hispanic Female')
plt.scatter(X_asian_female_test, y_asian_female_test, color='gold', label='Asian Female')

# Plotting the regression lines for each ethnicity and gender
plt.plot(X_white_male_test, y_white_male_pred, color='blue', linewidth=3)
plt.plot(X_black_male_test, y_black_male_pred, color='red', linewidth=3)
plt.plot(X_hispanic_male_test, y_hispanic_male_pred, color='green', linewidth=3)
plt.plot(X_asian_male_test, y_asian_male_pred, color='orange', linewidth=3)

plt.plot(X_white_female_test, y_white_female_pred, color='lightblue', linewidth=3)
plt.plot(X_black_female_test, y_black_female_pred, color='pink', linewidth=3)
plt.plot(X_hispanic_female_test, y_hispanic_female_pred, color='lightgreen', linewidth=3)
plt.plot(X_asian_female_test, y_asian_female_pred, color='gold', linewidth=3)

plt.xlabel('Year')
plt.ylabel('Deaths_per_100k_Resident_Population')
plt.title('Linear Regression Model by Ethnicity and Gender')
plt.legend()
plt.show()
