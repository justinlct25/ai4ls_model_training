from sklearn.cluster import KMeans
import folium
import pandas as pd

# Example dataset
#data = {
#    'Latitude': [47.5, 48.5, 49.5],
#    'Longitude': [16.5, 17.5, 18.5],
#    'pH': [3.0, 4.0, 5.0],
#    'CaCO3': [2.0, 2.5, 3.0]
#}
#df = pd.DataFrame(data)
data = pd.read_csv('./LUCAS-SOIL-2018(managed-l).csv')


# Perform clustering (here we use k-means as an example)
kmeans = KMeans(n_clusters=2)  # 2 clusters for managed and unmanaged land
features = soil_data_selected[['pH_CaCl2', 'pH_H2O', 'EC', 'OC', 'CaCO3', 'N', 'P', 'K']]  # Replace with the features you want to use for clustering
kmeans.fit(features)
soil_data_selected['Cluster'] = kmeans.labels_

# Ensure 'Cluster' column is of type int
soil_data_selected['Cluster'] = soil_data_selected['Cluster'].astype(int)

# Create a map object
m = folium.Map(location=[soil_data_selected['TH_LAT'].mean(), soil_data_selected['TH_LONG'].mean()], zoom_start=6)

# Color mapping for clusters
colors = ['red', 'blue']

# Add points to the map with cluster-based color
for idx, row in soil_data_selected.iterrows():
    cluster_index = int(row['Cluster'])  # Explicitly convert to int
    folium.CircleMarker(
        location=[row['TH_LAT'], row['TH_LONG']],
        radius=5,
        color=colors[cluster_index],
        fill=True,
        fill_color=colors[cluster_index],
        fill_opacity=0.6,
        popup=(f'Lat: {row["TH_LAT"]}, Lon: {row["TH_LONG"]}, '
               f'Cluster: {"Managed" if cluster_index == 0 else "Unmanaged"}')
    ).add_to(m)

# Display the map