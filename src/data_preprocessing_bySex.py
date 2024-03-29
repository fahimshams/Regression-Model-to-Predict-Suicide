#Load data
#Combine data
#Clean Data
#Handle missing values

import pandas as pd
pd.options.mode.chained_assignment = None

path_to_file = '../data/Suicide_Rates_In_US.xlsx'

data = pd.read_excel(path_to_file, sheet_name='By sex', engine='openpyxl')

df = pd.DataFrame(data)

df_subset = df[['UNIT', 'STUB_LABEL', 'YEAR', 'ESTIMATE']]

df_filtered = df_subset[(df_subset['UNIT'] == 'Deaths per 100,000 resident population, age-adjusted') & 
                        (df_subset['STUB_LABEL'].isin(['All persons','Male', 'Female']))]

column_names = {
    'UNIT': 'Measurement_Unit',
    'STUB_LABEL': 'Gender',
    'YEAR': 'Year',
    'ESTIMATE': 'Deaths_per_100k_Resident_Population'
}

df_filtered.rename(columns=column_names, inplace=True)
print(df_filtered.columns)


# Check for null values
null_values = df_filtered.isnull().sum()

# Display the columns with null values, if any
print("Columns with null values:")
print(null_values[null_values > 0])

for column in null_values[null_values > 0].index:
    if null_values[column] > 0:
        df_filtered[column].fillna(df_filtered[column].mean(), inplace=True)

print("Null values filled conditionally.")

output_path = '../data/Processed_Rates_By_Sex.xlsx'
sheet_name = 'By Sex'
df_filtered.to_excel(output_path, sheet_name=sheet_name, index=False)
print(f"DataFrame saved to '{output_path}' in sheet '{sheet_name}'.")
