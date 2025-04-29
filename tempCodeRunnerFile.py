ight'] = predicted_travel_times[idx]

    # Step 5: Save the weighted graph for use in route recommendation
    ox.save_graphml(tangier, os.path.join(base_path, 'tangier_weighted_network.graphml'))
    print("Weighted network saved. Ready for 