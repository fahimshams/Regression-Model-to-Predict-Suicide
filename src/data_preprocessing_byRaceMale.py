#Load data
#Combine data
#Clean Data
#Handle missing values

import pandas as pd
pd.options.mode.chained_assignment = None

path_to_file = '../data/Suicide_Rates_In_US.xlsx'

data = pd.read_excel(path_to_file, sheet_name='Male sex, race and Hispanic ori', engine='openpyxl')

df = pd.DataFrame(data)

df_subset = df[['UNIT', 'STUB_NAME', 'STUB_LABEL', 'YEAR', 'ESTIMATE']]

df_filtered = df_subset[(df_subset['UNIT'] == 'Deaths per 100,000 resident population, crude') & 
                        (df_subset['STUB_NAME'] == 'Sex and race and Hispanic origin') &
                        (df_subset['STUB_LABEL'].isin(['Male: Not Hispanic or Latino: White', 
                                                       'Male: Not Hispanic or Latino: Black or African American', 
                                                       'Male: Hispanic or Latino: All races',
                                                       'Male: Not Hispanic or Latino: Asian or Pacific Islander']))]

column_names = {
    'UNIT': 'Measurement_Unit',
    'STUB_LABEL': 'Race/Male',
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

output_path = '../data/Processed_Rates_By_Race_Male.xlsx'
sheet_name = 'Race'
df_filtered.to_excel(output_path, sheet_name=sheet_name, index=False)
print(f"DataFrame saved to '{output_path}' in sheet '{sheet_name}'.")
