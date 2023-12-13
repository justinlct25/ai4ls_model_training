import pandas as pd
import sys


if len(sys.argv) != 3:
    print("Usage: python script.py <original_csv_file> <2015_lucas_csv_file>")
    sys.exit(1)

original_csv_file = sys.argv[1]
textural_csv_file = sys.argv[2]

textural_types = ['Coarse', 'Clay', 'Sand', 'Silt']

original_df = pd.read_csv(original_csv_file)
textural_df = pd.read_csv(textural_csv_file)

textural_df = textural_df[['Point_ID'] + textural_types]

merged_df = pd.merge(original_df, textural_df, left_on='POINTID', right_on='Point_ID', how='left')
merged_df = merged_df.drop(columns=['Point_ID'])

merged_csv_file = original_csv_file.replace('.csv', '(textural-info).csv')
merged_df.to_csv(merged_csv_file, index=False)

# python3 csv-add-column-textural-info.py ./LUCAS-SOIL-2018\(managed-l\)\(bulk-density\)\(erosion\)\(out-standard\).csv ./LUCAS_Topsoil_2015_20200323.csv