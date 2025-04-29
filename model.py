import osmnx as ox
import networkx as nx
import pandas as pd
import geopandas as gpd
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
import numpy as np
from shapely.geometry import LineString, Point

def train_route_model():
    print("Loading data and training route recommendation model...")

    # Set the base path to the scripts directory
    base_path = os.path.abspath(os.path.dirname(__file__))

    # Load the street network from GraphML
    graph_path = os.path.join(base_path, 'tangier_network.graphml')
    tangier = ox.load_graphml(graph_path)

    # Load nodes and edges from JSON
    try:
        nodes = gpd.read_file(os.path.join(base_path, 'tangier_nodes.json'))
        edges = gpd.read_file(os.path.join(base_path, 'tangier_edges.json'))
    except Exception as e:
        print(f"Error loading JSON files: {str(e)}")
        return

    # Load amenities (optional) to identify green areas (e.g., parks)
    try:
        amenities = gpd.read_file(os.path.join(base_path, 'tangier_amenities.json')) if 'tangier_amenities.json' in os.listdir(base_path) else gpd.GeoDataFrame()
        # Filter for green areas (parks, gardens)
        green_areas = amenities[amenities['amenity'].isin(['park', 'garden'])] if not amenities.empty else gpd.GeoDataFrame()
    except:
        green_areas = gpd.GeoDataFrame()

    # Step 1: Calculate base edge weights based on distance, CO2, and greenness
    for u, v, key, data in tangier.edges(keys=True, data=True):
        # Distance (already in meters from OSM data)
        distance = data.get('length', 0)

        # CO2 emission factor based on road type
        highway = data.get('highway', 'unclassified')
        co2_factor = 1.0  # Default CO2 factor
        if highway in ['motorway', 'trunk']:
            co2_factor = 2.0  # High CO2 for highways (more car usage, higher speeds)
        elif highway in ['primary', 'secondary']:
            co2_factor = 1.5  # Moderate CO2 for major roads
        elif highway in ['residential', 'unclassified']:
            co2_factor = 1.0  # Standard CO2 for smaller roads
        elif highway in ['cycleway', 'path']:
            co2_factor = 0.2  # Low CO2 for bike/pedestrian paths
        co2_emission = distance * co2_factor  # Simplified: CO2 proportional to distance

        # Green route factor (favor bike paths, pedestrian paths, and routes near parks)
        green_factor = 1.0  # Default
        if highway in ['cycleway', 'path']:
            green_factor = 0.5  # Favor bike/pedestrian paths
        # Check if edge is near a green area (within ~100m, approximated as 0.001 degrees)
        if not green_areas.empty:
            edge_geom = LineString([(tangier.nodes[u]['x'], tangier.nodes[u]['y']), (tangier.nodes[v]['x'], tangier.nodes[v]['y'])])
            near_green = any(edge_geom.distance(pt) < 0.001 for pt in green_areas.geometry if pt is not None)
            if near_green:
                green_factor *= 0.8  # Reduce weight if near a park/garden

        # Composite weight: balance distance, CO2, and greenness
        # You can adjust these coefficients based on user preferences
        weight = (0.4 * distance) + (0.4 * co2_emission) + (0.2 * distance * green_factor)
        data['weight'] = weight

    # Step 2: Prepare features for ML model to predict travel time (optional enhancement)
    edge_features = pd.DataFrame({
        'length': [data['length'] for u, v, key, data in tangier.edges(keys=True, data=True)],
        'co2_emission': [data['length'] * (2.0 if data.get('highway', 'unclassified') in ['motorway', 'trunk'] else 
                         1.5 if data.get('highway', 'unclassified') in ['primary', 'secondary'] else 
                         1.0 if data.get('highway', 'unclassified') in ['residential', 'unclassified'] else 
                         0.2) for u, v, key, data in tangier.edges(keys=True, data=True)],
        'green_factor': [0.5 if data.get('highway', 'unclassified') in ['cycleway', 'path'] else 1.0 for u, v, key, data in tangier.edges(keys=True, data=True)]
    })

    # Adjust green_factor for proximity to green areas
    if not green_areas.empty:
        for idx, (u, v, key, data) in enumerate(tangier.edges(keys=True, data=True)):
            edge_geom = LineString([(tangier.nodes[u]['x'], tangier.nodes[u]['y']), (tangier.nodes[v]['x'], tangier.nodes[v]['y'])])
            near_green = any(edge_geom.distance(pt) < 0.001 for pt in green_areas.geometry if pt is not None)
            if near_green:
                edge_features.loc[idx, 'green_factor'] *= 0.8

    # Mock travel time (in seconds) with added noise
    np.random.seed(42)
    mock_travel_time = edge_features['length'] * 0.1
    noise = np.random.normal(0, 0.1, size=mock_travel_time.shape)
    mock_travel_time = mock_travel_time * (1 + noise)

    # Step 3: Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(edge_features, mock_travel_time, test_size=0.2, random_state=42)

    # Train a Linear Regression model to predict travel time
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Step 4: Evaluate the model's performance
    y_train_pred = model.predict(X_train)
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    print(f"Training Performance - Mean Squared Error: {train_mse:.2f}")
    print(f"Training Performance - R² Score: {train_r2:.2f}")

    y_test_pred = model.predict(X_test)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    print(f"Testing Performance - Mean Squared Error: {test_mse:.2f}")
    print(f"Testing Performance - R² Score: {test_r2:.2f}")

    # Cross-validation
    cv_scores = cross_val_score(model, edge_features, mock_travel_time, cv=5, scoring='r2')
    print(f"Cross-Validation R² Scores: {cv_scores}")
    print(f"Average Cross-Validation R²: {cv_scores.mean():.2f}")

    # Visualize predicted vs actual travel times (testing set)
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_test_pred, alpha=0.5, label='Predictions')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Ideal')
    plt.xlabel('Actual Travel Time (seconds)')
    plt.ylabel('Predicted Travel Time (seconds)')
    plt.title('Predicted vs Actual Travel Times (Testing Set)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(base_path, 'travel_time_predictions_test.png'), dpi=300)
    plt.close()

    # Step 5: Update edge weights using the ML model (optional)
    predicted_travel_times = model.predict(edge_features)
    for idx, (u, v, key, data) in enumerate(tangier.edges(keys=True, data=True)):
        # Optionally incorporate predicted travel time into the weight
        data['weight'] = (0.4 * data['length']) + (0.4 * edge_features.loc[idx, 'co2_emission']) + (0.2 * data['length'] * edge_features.loc[idx, 'green_factor'])

    # Step 6: Test a sample route
    start_coords = (35.7595, -5.8340)  # Near Tangier city center
    dest_coords = (35.7673, -5.7999)   # Near Tangier port
    start_node = ox.distance.nearest_nodes(tangier, start_coords[1], start_coords[0])
    dest_node = ox.distance.nearest_nodes(tangier, dest_coords[1], dest_coords[0])

    try:
        route = nx.shortest_path(tangier, start_node, dest_node, weight='weight')
        print("Sample route found successfully!")
    except nx.NetworkXNoPath:
        print("No route found between the sample coordinates.")
        return

    # Step 7: Visualize the sample route with green areas
    plt.figure(figsize=(10, 8))
    ox.plot_graph_route(tangier, route, route_color='r', route_linewidth=4, node_size=0, edge_linewidth=0.5)
    if not green_areas.empty:
        green_areas.plot(ax=plt.gca(), color='green', markersize=10, label='Green Areas (Parks/Gardens)')
    plt.title('Optimal Route in Tangier (Distance, CO2, Green)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(base_path, 'optimal_route.png'), dpi=300)
    plt.close()

    # Step 8: Save the weighted graph and model artifacts
    ox.save_graphml(tangier, os.path.join(base_path, 'tangier_optimal_network.graphml'))
    print("Weighted network saved. Ready for route recommendation.")

    # Save the trained model
    model_output_path = os.path.join(base_path, 'route_recommendation_model.pkl')
    joblib.dump(model, model_output_path)
    print(f"Model saved to {model_output_path}")

    # Save feature information for preprocessing
    feature_info = {
        'feature_names': edge_features.columns.tolist(),
        'feature_means': edge_features.mean().to_dict(),
        'feature_stds': edge_features.std().to_dict()
    }
    feature_info_path = os.path.join(base_path, 'feature_info.pkl')
    joblib.dump(feature_info, feature_info_path)
    print(f"Feature information saved to {feature_info_path}")

if __name__ == "__main__":
    train_route_model()
