import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load the dataset from Excel sheet
data = pd.read_excel('../data/Processed_Rates_By_Sex.xlsx', engine='openpyxl')

# Convert 'Year' column to datetime format
data['Year'] = pd.to_datetime(data['Year'], format='%Y')

# Time-Series Analysis
# Seasonality Detection
decomposition = seasonal_decompose(data.set_index('Year')['Deaths_per_100k_Resident_Population'], period=12)
seasonal = decomposition.seasonal

# Trend Analysis
trend = decomposition.trend

# Forecasting using SARIMA
train_data = data.set_index('Year')['Deaths_per_100k_Resident_Population']
model = SARIMAX(train_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
results = model.fit()
forecast = results.predict(start=len(train_data), end=len(train_data) + 5, dynamic=False)

# Gender-Based Analysis
male_data = data[data['Gender'] == 'Male']
female_data = data[data['Gender'] == 'Female']
all_person_data = data[data['Gender'] == 'All persons']

# Comparison
plt.plot(male_data['Year'], male_data['Deaths_per_100k_Resident_Population'], label='Male')
plt.plot(female_data['Year'], female_data['Deaths_per_100k_Resident_Population'], label='Female')
plt.plot(all_person_data['Year'], all_person_data['Deaths_per_100k_Resident_Population'], label='All Person')
plt.xlabel('Year')
plt.ylabel('Age-adjusted Suicide Rate (per 100,000) ')
plt.title('Suicide Rates in US by Gender 1950-2018')
plt.legend()
plt.show()

# Statistical Analysis
X = data[['Deaths_per_100k_Resident_Population']].values
y = data['Gender'].apply(lambda x: 0 if x == 'Male' else (1 if x == 'Female' else 2)).values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

classifier = RandomForestClassifier()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(classification_report(y_test, y_pred))
