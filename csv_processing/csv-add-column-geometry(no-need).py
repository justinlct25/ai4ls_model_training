
# import pandas as pd
# import geopandas as gpd


# # Read CSV and Shapefile
# original_df = pd.read_csv('./LUCAS-SOIL-2018(managed-l).csv')
# shapefile_path = "./LUCAS-SOIL-2018 .shp"
# gdf = gpd.read_file(shapefile_path)

# # Merge the CSV DataFrame with the GeoDataFrame based on POINTID
# merged_df = pd.merge(original_df, gdf, left_on='POINTID', right_on='POINTID')

# # Extract latitude and longitude from the geometry column
# merged_df['latitude'] = merged_df['geometry'].y
# merged_df['longitude'] = merged_df['geometry'].x

# # Drop the geometry column if you don't need it anymore
# merged_df = merged_df.drop(columns=['geometry'])

# # Save the result to a new CSV file
# merged_df.to_csv('./LUCAS-SOIL-2018(managed-l)(geometry).csv', index=False)



### NO NEED THIS THE EXCEL ALREADY HAVE GEOMETRY