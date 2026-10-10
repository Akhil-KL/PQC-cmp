"""
Blueprint-guided hole detection.

Uses the scale-invariant hole layout extracted from circular_top.png
to predict where holes should appear in the product photo, then
validates each predicted location against the actual image.

Blueprint-defined layout (scale-invariant ratios):
  Centre bore:  r/R ~ 0.24,  dist/R ~ 0.04  (centre of part)
  Bolt hole 1:  r/R ~ 0.057, dist/R ~ 0.90, angle ~ -135°
  Bolt hole 2:  r/R ~ 0.057, dist/R ~ 0.90, angle ~   30°
  Bolt hole 3:  r/R ~ 0.057, dist/R ~ 0.90, angle ~  152°

R = outer radius of the part as detected in the photo.

Run:
    python tools/blueprint_guided_detection.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np

# ── Blueprint-defined hole layout (scale-invariant) ──────────────────────
# Each entry: (name, r_ratio, dist_ratio, angle_deg, search_tolerance_r_ratio)
BLUEPRINT_HOLES = [
    # name,             r/R,    dist/R,  angle°,  tol_r/R
    ("centre_bore",     0.240,  0.04,    None,    0.06),   # angle=None = centre
    ("bolt_hole_1",     0.057,  0.90,    -135,    0.04),
    ("bolt_hole_2",     0.057,  0.90,      30,    0.04),
    ("bolt_hole_3",     0.057,  0.90,     152,    0.04),
]

# ── Load and preprocess photo ─────────────────────────────────────────────
img_path = Path("data/inputs/WhatsApp Image 2026-09-21 at 11.56.01 PM.jpeg")
orig = cv2.imread(str(img_path))
gray_orig = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)

# Quick preprocessing: find part boundary in photo
_, binary = cv2.threshold(gray_orig, 0, 255,
                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# Find the largest dark region (the part)
dark = cv2.bitwise_not(binary)
k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, k, iterations=3)
contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if not contours:
    print("Cannot find part boundary in photo")
    sys.exit(1)

main = max(contours, key=cv2.contourArea)
(part_cx, part_cy), part_r = cv2.minEnclosingCircle(main)
part_cx, part_cy, part_r = float(part_cx), float(part_cy), float(part_r)
print(f"Photo part: center=({part_cx:.0f},{part_cy:.0f})  R={part_r:.0f}px")

# ── For each blueprint hole, predict position and validate ────────────────
overlay = orig.copy()
cv2.circle(overlay, (int(part_cx), int(part_cy)), int(part_r), (0, 200, 0), 2)

results = []
for name, r_ratio, dist_ratio, angle_deg, tol in BLUEPRINT_HOLES:
    predicted_r = r_ratio * part_r
    search_r    = tol * part_r

    if angle_deg is None:
        # Centre hole
        pred_x = part_cx
        pred_y = part_cy
    else:
        rad = np.radians(angle_deg)
        pred_x = part_cx + dist_ratio * part_r * np.cos(rad)
        pred_y = part_cy + dist_ratio * part_r * np.sin(rad)

    # ── Validate: measure darkness at predicted location ──────────────
    ROI_R = int(predicted_r * 1.8)
    cx_i, cy_i = int(pred_x), int(pred_y)
    x1 = max(0, cx_i - ROI_R); x2 = min(gray_orig.shape[1], cx_i + ROI_R)
    y1 = max(0, cy_i - ROI_R); y2 = min(gray_orig.shape[0], cy_i + ROI_R)
    roi = gray_orig[y1:y2, x1:x2]

    # Interior circle mask
    h_roi, w_roi = roi.shape
    mask = np.zeros((h_roi, w_roi), np.uint8)
    cv2.circle(mask, (ROI_R, ROI_R), max(1, int(predicted_r * 0.85)), 255, -1)
    pix = roi[mask > 0]
    interior_mean = float(np.mean(pix)) if len(pix) > 0 else 255

    # Ring mask around the predicted hole (collar brightness)
    ring = np.zeros((h_roi, w_roi), np.uint8)
    cv2.circle(ring, (ROI_R, ROI_R), int(predicted_r * 1.5), 255, -1)
    cv2.circle(ring, (ROI_R, ROI_R), int(predicted_r * 1.0), 0, -1)
    ring = cv2.bitwise_and(ring, np.ones_like(ring) * 255)
    ring_pix = roi[ring > 0]
    collar_mean = float(np.mean(ring_pix)) if len(ring_pix) > 0 else 0

    contrast = abs(collar_mean - interior_mean)
    found = contrast > 10 or interior_mean < 70

    results.append({
        "name":        name,
        "pred_x":      round(pred_x),
        "pred_y":      round(pred_y),
        "pred_r":      round(predicted_r),
        "interior":    round(interior_mean, 1),
        "collar":      round(collar_mean, 1),
        "contrast":    round(contrast, 1),
        "found":       found,
    })

    # Draw on overlay
    col = (0, 0, 255) if found else (0, 100, 255)
    thick = 4 if found else 2
    cv2.circle(overlay, (cx_i, cy_i), int(predicted_r), col, thick)
    cv2.circle(overlay, (cx_i, cy_i), max(3, int(predicted_r) // 3), col, -1)
    label = f"{name.replace('_', ' ')} r={int(predicted_r)}"
    if not found:
        label += " MISSING?"
    cv2.putText(overlay, label,
                (cx_i + int(predicted_r) + 5, cy_i),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)

# ── Summary ───────────────────────────────────────────────────────────────
print("\n--- Blueprint-guided detection results ---")
print(f"{'Name':<20}  {'pred_r':>6}  {'int':>5}  {'collar':>6}  {'contrast':>8}  {'status'}")
print("-" * 70)
for r in results:
    status = "✓ FOUND" if r["found"] else "✗ MISSING"
    print(f"{r['name']:<20}  {r['pred_r']:6d}  {r['interior']:5.1f}  "
          f"{r['collar']:6.1f}  {r['contrast']:8.1f}  {status}")

n_found = sum(1 for r in results if r["found"])
print(f"\n{n_found}/{len(BLUEPRINT_HOLES)} expected holes detected")

# Save overlay
out = Path("outputs/phase2b_validation/blueprint_guided_overlay.jpg")
out.parent.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(out), overlay, [cv2.IMWRITE_JPEG_QUALITY, 92])
print(f"Overlay saved: {out}")
