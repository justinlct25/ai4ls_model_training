import pandas as pd

def get_summary_statistics(csv_file, columns):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file)

    # Select only the specified columns
    selected_columns = df[columns]

    # Initialize an empty list to store the results
    result = []

    # Iterate over each column
    for col in columns:
        try:
            # Try to convert the column to numeric values
            numeric_values = pd.to_numeric(selected_columns[col], errors='coerce')

            # If conversion is successful, calculate max, min, and average
            if not numeric_values.isnull().all():
                max_val = numeric_values.max()
                min_val = numeric_values.min()
                avg_val = numeric_values.mean()
                result.append({'Column': col, 'Max': max_val, 'Min': min_val, 'Avg': avg_val})
        except Exception as e:
            print(f"Error processing column {col}: {e}")

    # Convert the result to a DataFrame
    result_df = pd.DataFrame(result)

    return result_df

# Example usage
csv_file_path = './LUCAS-SOIL-2018.csv'
columns_to_check = ['pH_H2O', 'EC', 'OC', 'CaCO3', 'P', 'N', 'K']

result_dataframe = get_summary_statistics(csv_file_path, columns_to_check)

# Print the result
print(result_dataframe)
