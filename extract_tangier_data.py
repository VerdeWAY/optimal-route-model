import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString

# Set the user agent for the OSM API
ox.settings.log_console=True
ox.settings.use_cache=True
ox.settings.overpass_settings='[out:json][timeout:300]'

# Download the street network for Tangier
tangier = ox.graph_from_place('Tangier, Morocco', network_type='all')

# Save the street network as GraphML file
ox.save_graphml(tangier, 'tangier_network.graphml')

# Convert to GeoDataFrame for analysis
nodes, edges = ox.graph_to_gdfs(tangier)

# Save as shapefiles for use in GIS tools or further processing
nodes.to_file('tangier_nodes.shp')
edges.to_file('tangier_edges.shp')

# Extract different road types
drive_network = ox.graph_from_place('Tangier, Morocco', network_type='drive')
bike_network = ox.graph_from_place('Tangier, Morocco', network_type='bike')
walk_network = ox.graph_from_place('Tangier, Morocco', network_type='walk')

# Save these specific networks
ox.save_graphml(drive_network, 'tangier_drive_network.graphml')
ox.save_graphml(bike_network, 'tangier_bike_network.graphml')
ox.save_graphml(walk_network, 'tangier_walk_network.graphml')

# Visualize the network
fig, ax = plt.subplots(figsize=(12, 8))
ox.plot_graph(tangier, ax=ax, node_size=0, edge_linewidth=0.5)
plt.title('Tangier Road Network')
plt.tight_layout()
plt.savefig('tangier_network_map.png', dpi=300)

# Get additional data like amenities, public transportation stops
amenities = ox.features_from_place('Tangier, Morocco', {'amenity': True})
public_transport = ox.features_from_place('Tangier, Morocco', {'public_transport': True})

# Save these features
if not amenities.empty:
    amenities.to_file('tangier_amenities.shp')
if not public_transport.empty:
    public_transport.to_file('tangier_public_transport.shp')

print("Data extraction complete. Files saved in the current directory.")

# Optional: Print statistics about the network
print(f"Number of nodes: {len(tangier.nodes)}")
print(f"Number of edges: {len(tangier.edges)}")
print(f"Network type: {ox.get_digraph_to_undirected(tangier)}")