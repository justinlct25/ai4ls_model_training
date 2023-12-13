import pandas as pd
import sys


if len(sys.argv) != 3:
    print("Usage: python script.py <original_csv_file> <bulk_density_csv_file")
    sys.exit(1)

original_csv_file = sys.argv[1]
bulk_density_csv_file = sys.argv[2]

original_df = pd.read_csv(original_csv_file)
bulk_density_df = pd.read_csv(bulk_density_csv_file)

merged_df = pd.merge(original_df, bulk_density_df, left_on='POINTID', right_on='POINT_ID', how='left')
merged_df = merged_df.drop(columns=['POINT_ID'])

merged_csv_file = original_csv_file.replace('.csv', '(bulk-density).csv')
merged_df.to_csv(merged_csv_file, index=False)
