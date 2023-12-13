
import geopandas as gpd


# Provide the path to your Shapefile (.shp)
shapefile_path = "./LUCAS-SOIL-2018 .shp"

# Read the Shapefile
gdf = gpd.read_file(shapefile_path)

# Display the GeoDataFrame
print(gdf.head(100))
print(gdf.columns)


