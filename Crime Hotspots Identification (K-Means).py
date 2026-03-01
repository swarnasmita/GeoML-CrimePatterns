import pandas as pd
import folium
from sklearn.cluster import KMeans

# Load dataset
df = pd.read_csv('cyber_crime_with_zone1.csv')

# Coordinates mapping
state_coords = {
    'Andhra Pradesh': (15.9129, 79.7400), 'Arunachal Pradesh': (28.2180, 94.7278),
    'Assam': (26.2006, 92.9376), 'Bihar': (25.0961, 85.3131), 'Chhattisgarh': (21.2787, 81.8661),
    'Goa': (15.2993, 74.1240), 'Gujarat': (22.2587, 71.1924), 'Haryana': (29.0588, 76.0856),
    'Himachal Pradesh': (31.1048, 77.1734), 'Jammu & Kashmir': (33.7782, 76.5762),
    'Jharkhand': (23.6102, 85.2799), 'Karnataka': (15.3173, 75.7139), 'Kerala': (10.8505, 76.2711),
    'Madhya Pradesh': (22.9734, 78.6569), 'Maharashtra': (19.7515, 75.7139), 'Manipur': (24.6637, 93.9063),
    'Meghalaya': (25.4670, 91.3662), 'Mizoram': (23.1645, 92.9376), 'Nagaland': (26.1584, 94.5624),
    'Odisha': (20.9517, 85.0985), 'Punjab': (31.1471, 75.3412), 'Rajasthan': (27.0238, 74.2179),
    'Sikkim': (27.5330, 88.5122), 'Tamil Nadu': (11.1271, 78.6569), 'Telangana': (18.1124, 79.0193),
    'Tripura': (23.9408, 91.9882), 'Uttar Pradesh': (26.8467, 80.9462), 'Uttarakhand': (30.0668, 79.0193),
    'West Bengal': (22.9868, 87.8550), 'A & N Islands': (11.7401, 92.6586), 'Chandigarh': (30.7333, 76.7794),
    'D & N Haveli': (20.1809, 73.0169), 'Daman & Diu': (20.4283, 72.8397), 'Delhi': (28.7041, 77.1025),
    'Lakshadweep': (10.5667, 72.6417), 'Puducherry': (11.9416, 79.8083), 'Ladakh': (34.2268, 77.5619)
}

# Add coordinates to dataframe
df['Latitude'] = df['State/UT'].map(lambda x: state_coords.get(x, [None, None])[0])
df['Longitude'] = df['State/UT'].map(lambda x: state_coords.get(x, [None, None])[1])

# Loop for each year and perform clustering
for year in [2017, 2018, 2019, 2020]:
    year_df = df[df['Year'] == year].copy()

    # K-Means clustering based on 'Total'
    kmeans = KMeans(n_clusters=2, random_state=0)
    year_df['Cluster'] = kmeans.fit_predict(year_df[['Total']])

    # Identify which cluster is the "hotspot" one based on highest average crime
    cluster_means = year_df.groupby('Cluster')['Total'].mean()
    hotspot_cluster = cluster_means.idxmax()

    # Label hotspots
    year_df['Hotspot_KMeans'] = year_df['Cluster'].apply(lambda c: 1 if c == hotspot_cluster else 0)

    # Create folium map
    map_year = folium.Map(location=[22.9734, 78.6569], zoom_start=5, tiles="CartoDB positron")

    for _, row in year_df.iterrows():
        if pd.notnull(row['Latitude']) and pd.notnull(row['Longitude']):
            color = 'red' if row['Hotspot_KMeans'] == 1 else 'blue'
            folium.CircleMarker(
                location=[row['Latitude'], row['Longitude']],
                radius=8,
                popup=(
                    f"<b>{row['State/UT']}</b><br>"
                    f"Year: {year}<br>"
                    f"Total Crimes: {row['Total']}<br>"
                    f"K-Means Hotspot: {'Yes' if row['Hotspot_KMeans'] == 1 else 'No'}"
                ),
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7
            ).add_to(map_year)

    # Save map
    filename = f"cyber_crime_hotspots_kmeans_{year}.html"
    map_year.save(filename)
    print(f"✅ K-Means final hotspot map for {year} saved as '{filename}'")

    hotspot_states = year_df[year_df['Hotspot_KMeans'] == 1][['State/UT', 'Year', 'Total']]
    print(f"\n🔥 Hotspot States Detected by Kmeans Clustering for {year}:")
    if not hotspot_states.empty:
         print(hotspot_states.to_string(index=False))
    else:
        print("No hotspots detected for this year.")