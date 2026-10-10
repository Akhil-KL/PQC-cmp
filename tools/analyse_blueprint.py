"""
Extract the structural circles from circular_top.png blueprint and
map them back to the product photo coordinate system.

Blueprint circles (from visual inspection of circular_top.png):
  - Large outer disc (~r=199 in bp coords)  → part silhouette  
  - Raised hub outer ring                   → shoulder collar
  - Raised hub inner ring                   → bore collar  
  - Centre bore                             → through-hole (smallest centre)
  - 3 bolt holes on rim                     → small circles at ~120° spacing

This script:
1. Extracts circles from the blueprint with careful thresholding
2. Identifies which are HOLES vs structural geometry lines
3. Prints expected hole positions as fractions of part radius
   (scale-invariant, for matching to any photo)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np

bp_path = Path("data/blueprints/circular_top.png")
bp = cv2.imread(str(bp_path))
gray = cv2.cvtColor(bp, cv2.COLOR_BGR2GRAY)
h, w = gray.shape

# ---- Find part boundary ------------------------------------------------
# Invert and threshold
dark = (gray < 180).astype(np.uint8) * 255
k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, k, iterations=2)

contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if not contours:
    print("No contours found")
    sys.exit(1)

# Largest contour = part outline
main = max(contours, key=cv2.contourArea)
(part_cx, part_cy), part_r = cv2.minEnclosingCircle(main)
print(f"Part outline: center=({part_cx:.0f},{part_cy:.0f})  r={part_r:.0f}")
print(f"Blueprint size: {w}x{h}")

# ---- Find all circles with strong Hough --------------------------------
blurred = cv2.GaussianBlur(gray, (3, 3), 0)
raw = cv2.HoughCircles(
    blurred, cv2.HOUGH_GRADIENT,
    dp=1, minDist=20,
    param1=80, param2=25,
    minRadius=5, maxRadius=int(part_r * 0.85),
)

if raw is None:
    print("No Hough circles")
    sys.exit(1)

circles = np.round(raw[0, :]).astype(int)
print(f"\nTotal Hough circles: {len(circles)}")

# ---- Classify circles --------------------------------------------------
# A filled small circle = hole (dark interior surrounded by dark ring)
# A ring line = structural geometry (thin dark line, light interior)

results = []
for cx, cy, r in circles:
    # Distance from part centre
    dist_from_centre = np.sqrt((cx - part_cx)**2 + (cy - part_cy)**2)
    if dist_from_centre > part_r * 1.05:
        continue   # outside part

    # Measure interior brightness
    interior_mask = np.zeros(gray.shape, np.uint8)
    cv2.circle(interior_mask, (cx, cy), max(1, r - 3), 255, -1)
    ring_mask = np.zeros(gray.shape, np.uint8)
    cv2.circle(ring_mask, (cx, cy), r + 5, 255, -1)
    cv2.circle(ring_mask, (cx, cy), max(1, r - 3), 0, -1)

    interior_px = gray[interior_mask > 0]
    ring_px = gray[ring_mask > 0]

    if len(interior_px) == 0:
        continue

    int_mean = float(np.mean(interior_px))
    ring_mean = float(np.mean(ring_px)) if len(ring_px) > 0 else 255

    # Blueprint holes = small circles with dark interior (drawn as solid dots
    # or hollow circles with white interior surrounded by thin dark ring)
    # Blueprint geometry lines = thin dark rings with white (light) interior

    is_hole = int_mean < 200   # dark interior = hole marker in blueprint
    dist_ratio = dist_from_centre / part_r

    results.append({
        "cx": cx, "cy": cy, "r": r,
        "dist_from_centre": dist_from_centre,
        "dist_ratio": round(dist_ratio, 3),
        "r_ratio": round(r / part_r, 3),
        "int_mean": round(int_mean, 1),
        "is_hole": is_hole,
    })

# Sort by radius descending
results.sort(key=lambda x: -x["r"])

print("\n--- All circles inside part (sorted by radius) ---")
print(f"{'r':>5}  {'r/R':>5}  {'dist/R':>6}  {'int':>5}  {'hole?':>6}  center")
for row in results[:40]:
    print(f"{row['r']:5d}  {row['r_ratio']:5.3f}  {row['dist_ratio']:6.3f}  "
          f"{row['int_mean']:5.1f}  {'YES' if row['is_hole'] else 'no':>6}  "
          f"({row['cx']},{row['cy']})")

# ---- Identify key structural circles -----------------------------------
print("\n--- KEY STRUCTURAL CIRCLES ---")
# Part outer boundary
print(f"Part outer radius:   r={part_r:.0f}px  (reference = 1.0)")

# Holes = small, dark interior, inside the part
holes_found = [r for r in results if r["is_hole"] and r["r"] < part_r * 0.15]
print(f"\nHole candidates ({len(holes_found)}):")
for h in sorted(holes_found, key=lambda x: x["r"], reverse=True):
    print(f"  r={h['r']}px ({h['r_ratio']:.3f}×R)  "
          f"dist={h['dist_ratio']:.3f}×R  "
          f"int_brightness={h['int_mean']:.0f}  "
          f"pos=({h['cx']},{h['cy']})")

# Scale-invariant hole summary
print("\n--- SCALE-INVARIANT HOLE SUMMARY ---")
print("(ratios relative to part outer radius R)")
for h in sorted(holes_found, key=lambda x: x["r"], reverse=True):
    angle = np.degrees(np.arctan2(h["cy"] - part_cy, h["cx"] - part_cx))
    print(f"  r/R={h['r_ratio']:.3f}  dist/R={h['dist_ratio']:.3f}  "
          f"angle={angle:.0f}°")
