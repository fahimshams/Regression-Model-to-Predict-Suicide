#Load data
#Combine data
#Clean Data
#Handle missing values

import pandas as pd
pd.options.mode.chained_assignment = None

path_to_file = '../data/Suicide_Rates_In_US.xlsx'

data = pd.read_excel(path_to_file, sheet_name='Female Age group', engine='openpyxl')

df = pd.DataFrame(data)

df_subset = df[['UNIT', 'STUB_NAME', 'STUB_LABEL', 'YEAR', 'ESTIMATE']]

df_filtered = df_subset[df_subset['STUB_LABEL'].isin(['Female: 10-14 years', 'Female: 15-24 years', 'Female: 25-34 years', 'Female: 35-44 years', 'Female: 45-64 years', 'Female: 65 years and over'])]


column_names = {
    'UNIT': 'Measurement_Unit',
    'STUB_LABEL': 'Age_Group',
    'YEAR': 'Year',
    'ESTIMATE': 'Deaths_per_100k_Resident_Population'
}

df_filtered.rename(columns=column_names, inplace=True)
print(df_filtered.columns)


# Check for null values
null_values = df_filtered.isnull()

# Display the columns with null values, if any
print("Columns with null values:")
print(null_values[null_values > 0].index)

for column_name in null_values.columns[null_values.any()]:
    df_filtered[column_name].fillna(df_filtered[column_name].mean(), inplace=True)


print("Null values filled conditionally.")

output_path = '../data/Processed_Rates_By_Age_Female.xlsx'
sheet_name = 'Age'
df_filtered.to_excel(output_path, sheet_name=sheet_name, index=False)
print(f"DataFrame saved to '{output_path}' in sheet '{sheet_name}'.")
