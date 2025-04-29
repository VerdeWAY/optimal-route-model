import geopandas as gpd
import json
import os

def convert_shp_to_json():
    print("Converting shapefiles to JSON...")

    # Set the base path to the scripts directory (where the script is running)
    base_path = os.path.abspath(os.path.dirname(__file__))

    # List of shapefiles to convert (relative to the scripts directory)
    shapefiles = [
        'tangier_nodes.shp',
        'tangier_edges.shp',
        'tangier_amenities.shp',
        'tangier_public_transport.shp'
    ]

    for shp_file in shapefiles:
        shp_path = os.path.join(base_path, shp_file)
        try:
            # Load the shapefile
            gdf = gpd.read_file(shp_path)

            # Drop NaN values to avoid serialization issues
            gdf = gdf.dropna()

            # Create the output JSON filename (replace .shp with .json)
            json_file = shp_file.replace('.shp', '.json')
            json_path = os.path.join(base_path, json_file)

            # Convert GeoDataFrame to JSON string (remove orient parameter)
            json_str = gdf.to_json(na='drop')

            # Write the JSON string to a file
            with open(json_path, 'w') as f:
                f.write(json_str)

            print(f"Converted {shp_file} to {json_file}")
        except FileNotFoundError:
            print(f"Shapefile {shp_file} not found at {shp_path}. Skipping...")
        except Exception as e:
            print(f"Error converting {shp_file}: {str(e)}")

if __name__ == "__main__":
    convert_shp_to_json()