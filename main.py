"""
Satellite-Based Maritime Intelligence System


Pipeline:
    Satellite Image → YOLOv8 Detection → Ship Counting → Congestion Analysis
    → Military/Civilian Classification → Risk Assessment → Alert System
    → Density Heatmap → Zone-Based Hotspot Detection
"""


import argparse


from analysis.maritime_analysis import (
    get_ship_counts,
    congestion_level,
    classify_military_civilian,
    risk_level,
    check_unusual_clustering,
    alert_system,
    density_label
)

from visualization.visualizer import (
    generate_heatmap,
    zone_based_analysis,
    show_detection_results
)


from models.detector import (
    load_model,
    run_detection
)



# Main Pipeline


def run_pipeline(image_path, weights_path, grid_size=4):
    print("\n" + "=" * 55)
    print("   Satellite Maritime Intelligence System")
    print("=" * 55)

    # Load model
    try:
        print(f"\n[*] Loading model from: {weights_path}")
        model = load_model(weights_path)

        # Run detection
        print(f"[*] Running detection on: {image_path}")
        results = run_detection(model, image_path)

    except Exception as e:
        print(f"[ERROR] {e}")
        return    

    

    show_detection_results(results)

    orig_img = results[0].orig_img.copy()

    # Ship Counting 
    total_ships, named_counts, classes = get_ship_counts(results, model)
    print(f"\n[Ship Count]")
    print(f"  Total Ships     : {total_ships}")
    print(f"  Class Breakdown : {named_counts}")

    #  Congestion
    congestion = congestion_level(total_ships)
    print(f"\n[Congestion]")
    print(f"  Traffic Level   : {congestion}")

    density = density_label(total_ships)

    print(f"  Traffic Density : {density}")

    #  Military / Civilian Classification
    military_count, civilian_count, unknown_count = (
    classify_military_civilian(classes, model))
    print(f"\n[Risk Assessment]")
    print(f"  Military Ships  : {military_count}")
    print(f"  Civilian Ships  : {civilian_count}")
    print(f"  Unknown Ships   : {unknown_count}")
    print(f"  Risk Level      : {risk_level(military_count, total_ships)}")

    # Unusual Clustering
    is_clustered, cluster_msg = check_unusual_clustering(results)
    print(f"\n[Clustering Analysis]")
    print(f"  {'[ALERT] ' if is_clustered else ''}{cluster_msg}")

    # Alert System 
    alert = alert_system(military_count, total_ships, congestion)
    print(f"\n[System Alert]")
    print(f"  {alert}")

    # Density Heatmap
    print(f"\n[*] Generating density heatmap...")
    generate_heatmap(results, orig_img)

    # Zone-Based Hotspot Detection
    print(f"[*] Running zone-based hotspot detection ({grid_size}x{grid_size} grid)...")
    zone_counts, hotspots, zone_overlay = (
    zone_based_analysis(results, orig_img, grid_size))

    print(f"\n[Zone Analysis]")
    print(f"  Zone Ship Counts:\n{zone_counts}")
    if len(hotspots) > 0:
        print(f"  Hotspot Zones (row, col): {hotspots.tolist()}")
    else:
        print("  No hotspots detected.")

    print("\n" + "=" * 55)
    print("   Analysis Complete")
    print("=" * 55 + "\n")



# Entry Point


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Maritime Intelligence System")
    parser.add_argument(
        "--image", type=str, required=True,
        help="Path to input satellite image (e.g. test/images/sample.jpg)"
    )
    parser.add_argument(
        "--weights", type=str, default="runs/detect/train/weights/best.pt",
        help="Path to trained YOLOv8 weights (default: runs/detect/train/weights/best.pt)"
    )
    parser.add_argument(
        "--grid", type=int, default=4,
        help="Grid size for zone-based hotspot detection (default: 4)"
    )
    args = parser.parse_args()

    run_pipeline(
        image_path=args.image,
        weights_path=args.weights,
        grid_size=args.grid
    )
