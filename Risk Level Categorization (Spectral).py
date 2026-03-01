import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import SpectralClustering
from sklearn.neighbors import kneighbors_graph
import seaborn as sns
import networkx as nx
import folium
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('cyber_crime_with_zone1.csv')

# State coordinates
state_coordinates = {
    'Andhra Pradesh': (15.9129, 79.7400), 'Arunachal Pradesh': (28.2180, 94.7278),
    'Assam': (26.2006, 92.9376), 'Bihar': (25.0961, 85.3131), 'Chhattisgarh': (21.2787, 81.8661),
    'Goa': (15.2993, 74.1240), 'Gujarat': (22.2587, 71.1924), 'Haryana': (29.0588, 76.0856),
    'Himachal Pradesh': (31.1048, 77.1734), 'Jammu & Kashmir': (33.7782, 76.5762),
    'Jharkhand': (23.6102, 85.2799), 'Karnataka': (15.3173, 75.7139), 'Kerala': (10.8505, 76.2711),
    'Madhya Pradesh': (22.9734, 78.6569), 'Maharashtra': (19.7515, 75.7139),
    'Manipur': (24.6637, 93.9063), 'Meghalaya': (25.4670, 91.3662), 'Mizoram': (23.1645, 92.9376),
    'Nagaland': (26.1584, 94.5624), 'Odisha': (20.9517, 85.0985), 'Punjab': (31.1471, 75.3412),
    'Rajasthan': (27.0238, 74.2179), 'Sikkim': (27.5330, 88.5122), 'Tamil Nadu': (11.1271, 78.6569),
    'Telangana': (18.1124, 79.0193), 'Tripura': (23.9408, 91.9882), 'Uttar Pradesh': (26.8467, 80.9462),
    'Uttarakhand': (30.0668, 79.0193), 'West Bengal': (22.9868, 87.8550),
    'A & N Islands': (11.7401, 92.6586), 'Chandigarh': (30.7333, 76.7794),
    'D & N Haveli': (20.1809, 73.0169), 'Daman & Diu': (20.4283, 72.8397),
    'Delhi': (28.7041, 77.1025), 'Lakshadweep': (10.5667, 72.6417),
    'Puducherry': (11.9416, 79.8083),
    'Ladakh': (34.2268, 77.5619)
}

# Add coordinates
df['Latitude'] = df['State/UT'].map(lambda x: state_coordinates.get(x, (np.nan, np.nan))[0])
df['Longitude'] = df['State/UT'].map(lambda x: state_coordinates.get(x, (np.nan, np.nan))[1])
df.dropna(subset=['Latitude', 'Longitude', 'Normalized Crime Rate', 'Year'], inplace=True)

# Analyze for each year
for year in [2017, 2018, 2019, 2020]:
    print(f"\n📌 Processing year {year}")
    year_df = df[df['Year'] == year].copy()

    # Prepare features
    features = year_df[['Latitude', 'Longitude', 'Normalized Crime Rate']]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # Create affinity matrix using KNN graph
    knn_graph = kneighbors_graph(scaled_features, n_neighbors=5, include_self=True)
    affinity_matrix = knn_graph.toarray()

    # Spectral Clustering
    spectral = SpectralClustering(n_clusters=4, affinity='precomputed', random_state=42)
    year_df['Spectral_Cluster'] = spectral.fit_predict(affinity_matrix)

    # Assign risk levels
    cluster_means = year_df.groupby('Spectral_Cluster')['Normalized Crime Rate'].mean().sort_values()
    risk_labels = {cluster: label for cluster, label in zip(cluster_means.index,
                   ['Low Risk', 'Moderate Risk', 'High Risk', 'Very High Risk'])}
    year_df['Risk_Level'] = year_df['Spectral_Cluster'].map(risk_labels)

    # Risk level colors
    risk_colors = {
        'Low Risk': 'green', 'Moderate Risk': 'orange',
        'High Risk': 'red', 'Very High Risk': 'darkred'
    }
    year_df['Color'] = year_df['Risk_Level'].map(risk_colors)

    # Save map
    m = folium.Map(location=[22.9734, 78.6569], zoom_start=5, tiles="CartoDB positron")
    for _, row in year_df.iterrows():
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=6 + row['Normalized Crime Rate'] * 0.8,
            popup=(f"<b>{row['State/UT']}</b><br>"
                   f"Zone: {row['Zone']}<br>"
                   f"Crime Rate: {row['Normalized Crime Rate']:.2f}<br>"
                   f"Cluster: {row['Spectral_Cluster']}<br>"
                   f"<b>Risk Level: {row['Risk_Level']}</b>"),
            color=row['Color'], fill=True,
            fill_color=row['Color'], fill_opacity=0.7
        ).add_to(m)
    m.save(f"spectral_crime_hotspots_{year}.html")
    print(f"✅ Spectral clustering map saved as spectral_crime_hotspots_{year}.html")

    # 🟦 Affinity Matrix Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(affinity_matrix, cmap='coolwarm', linewidths=0.5)
    plt.title(f"Affinity Matrix Heatmap - {year}")
    plt.xlabel("State Index")
    plt.ylabel("State Index")
    plt.show()

    # 🔗 Similarity Graph
    G = nx.from_numpy_array(affinity_matrix)
    G = nx.relabel_nodes(G, dict(zip(range(len(year_df)), year_df['State/UT'])))
    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(G, seed=42)
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color='skyblue')
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.6)
    nx.draw_networkx_labels(G, pos, font_size=9)
    plt.title(f"Similarity Graph (Spectral Affinity) - {year}")
    plt.axis('off')
    plt.show()

    print(f"\n📌 States Grouped by Risk Level for {year}:")
    for level, states in year_df.groupby('Risk_Level')['State/UT']:
        print(f"{level}:")
        print(", ".join(states))
        print()