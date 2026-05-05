"""
Satellite-Based Maritime Intelligence System


Pipeline:
    Satellite Image → YOLOv8 Detection → Ship Counting → Congestion Analysis
    → Military/Civilian Classification → Risk Assessment → Alert System
    → Density Heatmap → Zone-Based Hotspot Detection
"""

import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from collections import Counter
from scipy.spatial.distance import pdist
from ultralytics import YOLO



# Ship class definitions


MILITARY_CLASSES = [
    "Destroyer", "Cruiser", "Frigate", "Submarine",
    "Warship", "Commander", "Aircraft Carrier"
]

CIVILIAN_CLASSES = [
    "Cargo", "Tanker", "Fishing", "Passenger", "Yacht", "Barge",
    "Container Ship", "Bulk Carrier", "Oil Tanker", "Tug",
    "Auxiliary Ships", "Patrol", "Landing", "Hovercraft",
    "Sailboat", "Carrier", "Boat", "Other"
]

ALL_SHIP_CLASSES = [
    "Aircraft Carrier", "Auxiliary Ships", "Barge", "Cargo", "Commander",
    "Container Ship", "Cruiser", "Destroyer", "Frigate", "Landing",
    "Passenger", "Patrol", "Submarine", "Tanker", "Tug", "Warship",
    "Yacht", "Fishing", "Boat", "Hovercraft", "Sailboat", "Carrier",
    "Bulk Carrier", "Oil Tanker", "Other"
]



# Analysis Functions


def get_ship_counts(results, model):
    """Count total ships and break down by class name."""
    boxes = results[0].boxes
    classes = boxes.cls.cpu().numpy()
    total = len(classes)
    class_counts = Counter(classes)
    named_counts = {model.names[int(k)]: v for k, v in class_counts.items()}
    return total, named_counts, classes


def congestion_level(count):
    """Classify traffic congestion based on total ship count."""
    if count < 5:
        return "Low"
    elif count < 15:
        return "Medium"
    else:
        return "High"


def classify_military_civilian(classes, model):
    """Split detected ships into military and civilian counts."""
    military = 0
    civilian = 0
    for cls in classes:
        name = model.names[int(cls)]
        if name in MILITARY_CLASSES:
            military += 1
        else:
            civilian += 1
    return military, civilian


def risk_level(military, total):
    """
    Assess risk based on military ship ratio.
        Low    → no military ships
        Medium → military ships < 30% of total
        High   → military ships >= 30% of total
    """
    if military == 0:
        return "Low"
    elif military < total * 0.3:
        return "Medium"
    else:
        return "High"


def check_unusual_clustering(results, threshold=50):
    """
    Detect if ships are suspiciously close together.
    Computes pairwise distances between ship centers.
    """
    centers = []
    for box in results[0].boxes.xyxy:
        x1, y1, x2, y2 = box.cpu().numpy()
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        centers.append([cx, cy])

    centers = np.array(centers)

    if len(centers) < 2:
        return False, "Not enough ships to check clustering."

    distances = pdist(centers)
    if np.min(distances) < threshold:
        return True, f"Ships are closely clustered (min distance: {np.min(distances):.1f}px)"
    return False, "Normal distribution."


def alert_system(military_count, total_ships, congestion):
    """Generate a system alert based on current maritime situation."""
    if military_count > 0 and congestion == "High":
        return "CRITICAL ALERT: Military ships in high congestion zone!"
    elif military_count > 0:
        return "WARNING: Military presence detected."
    elif congestion == "High":
        return "WARNING: Heavy maritime traffic."
    else:
        return "Normal maritime activity."



# Visualization Functions


def generate_heatmap(results, orig_img):
    """Overlay a Gaussian density heatmap on the original image."""
    h, w, _ = orig_img.shape
    heatmap = np.zeros((h, w), dtype=np.float32)

    for box in results[0].boxes.xyxy:
        x1, y1, x2, y2 = map(int, box.cpu().numpy())
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        cv2.circle(heatmap, (cx, cy), 50, 1, -1)

    heatmap = cv2.GaussianBlur(heatmap, (51, 51), 0)

    if heatmap.max() > 0:
        heatmap /= heatmap.max()

    heatmap_color = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(orig_img, 0.6, heatmap_color, 0.4, 0)

    plt.figure(figsize=(8, 6))
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("Ship Density Heatmap")
    plt.tight_layout()
    plt.show()


def zone_based_analysis(results, orig_img, grid_size=4):
    """
    Divide image into a grid and count ships per zone.
    Highlights zones by density and identifies hotspots.
    Returns zone_counts array and hotspot coordinates.
    """
    h, w, _ = orig_img.shape
    zone_h = h // grid_size
    zone_w = w // grid_size

    zone_counts = np.zeros((grid_size, grid_size), dtype=int)

    # Assign each ship to a grid zone
    for box in results[0].boxes.xyxy:
        x1, y1, x2, y2 = map(int, box.cpu().numpy())
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        grid_x = min(cx // zone_w, grid_size - 1)
        grid_y = min(cy // zone_h, grid_size - 1)

        zone_counts[grid_y][grid_x] += 1

    # Draw zone overlay
    overlay = orig_img.copy()
    for i in range(grid_size):
        for j in range(grid_size):
            x1 = j * zone_w
            y1 = i * zone_h
            x2 = x1 + zone_w
            y2 = y1 + zone_h
            count = zone_counts[i][j]

            # Color: green → yellow → red based on ship count
            if count == 0:
                color = (0, 255, 0)
            elif count < 3:
                color = (0, 255, 255)
            else:
                color = (0, 0, 255)

            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            cv2.putText(overlay, str(count), (x1 + 10, y1 + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    plt.figure(figsize=(8, 6))
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("Zone-Based Traffic Map")
    plt.tight_layout()
    plt.show()

    # Find hotspot zones
    max_count = np.max(zone_counts)
    hotspots = np.argwhere(zone_counts == max_count) if max_count > 0 else []

    return zone_counts, hotspots



# Main Pipeline


def run_pipeline(image_path, weights_path, grid_size=4):
    print("\n" + "=" * 55)
    print("   Satellite Maritime Intelligence System")
    print("=" * 55)

    # Load model
    print(f"\n[*] Loading model from: {weights_path}")
    model = YOLO(weights_path)

    # Run detection
    print(f"[*] Running detection on: {image_path}")
    results = model(image_path)

    # Show detection result
    img = results[0].plot()
    plt.figure(figsize=(10, 7))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("YOLOv8 Ship Detection")
    plt.tight_layout()
    plt.show()

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

    #  Traffic Density Label
    if total_ships < 5:
        density_label = "Sparse"
    elif total_ships < 15:
        density_label = "Moderate"
    else:
        density_label = "Dense"
    print(f"  Traffic Density : {density_label}")

    #  Military / Civilian Classification
    military_count, civilian_count = classify_military_civilian(classes, model)
    print(f"\n[Risk Assessment]")
    print(f"  Military Ships  : {military_count}")
    print(f"  Civilian Ships  : {civilian_count}")
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
    zone_counts, hotspots = zone_based_analysis(results, orig_img, grid_size)

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
