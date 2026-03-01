import pandas as pd
import folium
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import SpectralClustering
from sklearn.neighbors import kneighbors_graph
import seaborn as sns
import numpy as np

# Load dataset
df = pd.read_csv('cyber_crime_with_zone1.csv')

# Selected crime features
selected_features = [
    'Personal Revenge', 'Anger', 'Fraud', 'Extortion', 'Causing Disrepute', 'Prank',
    'Sexual Exploitation', 'Political Motives', 'Terrorist Activities', 'Inciting Hate against Country',
    'Disrupt Public Service', 'Sale Purchase Illegal Drugs', 'Developing own Business', 'Spreading Piracy',
    'Psycho or Pervert', 'Steal Information', 'Abetment to Suicide', 'Others'
]

# Coordinates dictionary
state_coordinates = {
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

# Crime color mapping
crime_colors = {
    'Personal Revenge': 'blue', 'Anger': 'red', 'Fraud': 'orange', 'Extortion': 'purple',
    'Causing Disrepute': 'green', 'Prank': 'yellow', 'Sexual Exploitation': 'pink',
    'Political Motives': 'brown', 'Terrorist Activities': 'darkblue',
    'Inciting Hate against Country': 'violet', 'Disrupt Public Service': 'cyan',
    'Sale Purchase Illegal Drugs': 'darkgreen', 'Developing own Business': 'gold',
    'Spreading Piracy': 'gray', 'Psycho or Pervert': 'darkred', 'Steal Information': 'lightblue',
    'Abetment to Suicide': 'lightgreen', 'Others': 'lightgray'
}

# Group and sum data by year/state
df_grouped = df.groupby(['Year', 'State/UT'])[selected_features].sum().reset_index()

# Loop over each year
for year in sorted(df_grouped['Year'].unique()):
    print(f"\n📊 Clustering for Year {year}")

    year_data = df_grouped[df_grouped['Year'] == year].copy()

    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(year_data[selected_features])

    # ➕ Construct adjacency matrix using nearest neighbors
    adj_matrix = kneighbors_graph(features_scaled, n_neighbors=15, mode='connectivity', include_self=False)
    A = adj_matrix.toarray()

    # ➕ Degree matrix
    D = np.diag(A.sum(axis=1))

    # ➕ Graph Laplacian matrix: L = D - A
    L = D - A

    # 🔥 Visualize Adjacency Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(A, cmap='Blues')
    plt.title(f'Adjacency Matrix (A) - Year {year}')
    plt.xlabel('States')
    plt.ylabel('States')
    plt.tight_layout()
    plt.show()

    # 🔥 Visualize Laplacian Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(L, cmap='Reds')
    plt.title(f'Graph Laplacian Matrix (L = D - A) - Year {year}')
    plt.xlabel('States')
    plt.ylabel('States')
    plt.tight_layout()
    plt.show()

    # Apply Spectral Clustering
    spectral = SpectralClustering(n_clusters=3, affinity='nearest_neighbors', n_neighbors=15, random_state=42)
    year_data['Cluster'] = spectral.fit_predict(features_scaled)

    # Dominant crime type per state
    year_data['Dominant_Crime'] = year_data[selected_features].idxmax(axis=1)

    # Folium map
    m = folium.Map(location=[22.9734, 78.6569], zoom_start=5, tiles='CartoDB positron')
    for _, row in year_data.iterrows():
        state = row['State/UT']
        dominant_crime = row['Dominant_Crime']
        crime_color = crime_colors.get(dominant_crime, 'lightgray')
        coords = state_coordinates.get(state)
        if coords:
            folium.CircleMarker(
                location=coords,
                radius=7,
                color=crime_color,
                fill=True,
                fill_color=crime_color,
                fill_opacity=0.7,
                popup=folium.Popup(
                    f"<b>{state}</b><br>Cluster: {row['Cluster']}<br>Dominant Crime: <i>{dominant_crime}</i>",
                    max_width=250
                )
            ).add_to(m)

    # Save map
    map_file = f"crime_spectral_clusters_{year}.html"
    m.save(map_file)
    print(f"✅ Saved map as {map_file}")

    # Plotting state distribution across clusters
    plt.figure(figsize=(10, 5))
    sns.countplot(data=year_data, x='Cluster', palette='Set2')
    plt.title(f"State Distribution Across Clusters - {year}")
    plt.xlabel("Cluster Label")
    plt.ylabel("Number of States")
    plt.tight_layout()
    plt.show()

    # Plotting dominant crime type distribution per cluster
    plt.figure(figsize=(12, 6))
    sns.countplot(data=year_data, x='Dominant_Crime', hue='Cluster', palette='Set2')
    plt.xticks(rotation=45, ha='right')
    plt.title(f"Dominant Crime Types per Cluster - {year}")
    plt.xlabel("Crime Type")
    plt.ylabel("Number of States")
    plt.tight_layout()
    plt.show()


    # 🔽 Print states with dominant crime types
    print(f"🗺️ Dominant Crime Types by State in {year}:")
    for _, row in year_data.iterrows():
        print(f" - {row['State/UT']}: {row['Dominant_Crime']}")