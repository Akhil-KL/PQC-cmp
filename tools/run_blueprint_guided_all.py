"""
Blueprint-guided hole/feature detection on all product images.

Automatically identifies which of the two known parts is in the photo,
then predicts where holes/features should appear using scale-invariant
ratios extracted from the blueprints, and validates each location.

Two supported parts:
  1. CIRCULAR DISC  → data/blueprints/circular_top.png
       - Centre bore     r/W ≈ 0.240, dist/R ≈ 0.04 (centre)
       - Bolt hole ×3    r/W ≈ 0.057, dist/R ≈ 0.90, 120° apart
       Part identified by: enclosing-circle fill ratio > 0.75

  2. BRACKET (box_front) → data/blueprints/box_front.png
       - Small bolt hole top     rel=(+0.13, -0.27), r/W ≈ 0.068
       - Small bolt hole bottom  rel=(+0.13, -0.11), r/W ≈ 0.068
       - Large boss/through-hole rel=(+0.07, +0.36), r/W ≈ 0.102
       - Square cutout           rel=(-0.25,  0.00), w/W ≈ 0.50, h/H ≈ 0.30
       Part identified by: enclosing-circle fill ratio < 0.55

Usage:
    python tools/run_blueprint_guided_all.py
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np

INPUT_DIR  = Path("data/inputs")
OUTPUT_DIR = Path("outputs/blueprint_guided")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# ── Part layouts ─────────────────────────────────────────────────────────────
# CIRCULAR: ratios relative to outer radius R
CIRCULAR_HOLES = [
    # name,           r/R,    dist/R,  angle°,  tol/R
    ("centre_bore",   0.240,  0.040,   None,    0.06),
    ("bolt_hole_1",   0.057,  0.900,   -135,    0.05),
    ("bolt_hole_2",   0.057,  0.900,     30,    0.05),
    ("bolt_hole_3",   0.057,  0.900,    152,    0.05),
]

# BRACKET: ratios relative to part bounding-box width W and height H
# (rel_x, rel_y) are fractions of W and H from the part centre
BRACKET_FEATURES = [
    # name,                  rel_x,  rel_y,  r/W,    is_circle, tol/W
    ("bolt_hole_top",         0.129, -0.267,  0.068,  True,      0.05),
    ("bolt_hole_bottom",      0.129, -0.114,  0.068,  True,      0.05),
    ("boss_through_hole",     0.065,  0.362,  0.102,  True,      0.06),
    ("square_cutout",        -0.253,  0.000,  0.000,  False,     0.00),
]


# ── Part detector ─────────────────────────────────────────────────────────────
def detect_part(gray: np.ndarray):
    """
    Returns (part_type, part_cx, part_cy, part_r_or_w, part_h, contour).
    part_type: 'circular' | 'bracket' | None
    """
    _, binary = cv2.threshold(gray, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark = cv2.bitwise_not(binary)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, k, iterations=3)
    contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, None, None, None, None

    H, W = gray.shape
    BORDER = 25   # pixels — contours touching this margin are likely shadows/frame

    # Score each contour: prefer large, non-border, compact shapes
    best, best_score = None, -1.0
    for c in contours:
        area = cv2.contourArea(c)
        if area < 1500:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        # Skip if it touches the image border (shadow, table edge, frame)
        if x <= BORDER or y <= BORDER or x + bw >= W - BORDER or y + bh >= H - BORDER:
            continue
        (cx, cy), r = cv2.minEnclosingCircle(c)
        fill = area / (np.pi * r * r + 1e-6)
        # Favour large, well-filled contours that are away from the border
        score = area * fill
        if score > best_score:
            best_score = score
            best = c

    if best is None:
        # Fallback: largest contour regardless of border (last resort)
        best = max(contours, key=cv2.contourArea)

    area = cv2.contourArea(best)
    if area < 1500:
        return None, None, None, None, None, None

    (cx, cy), enclosing_r = cv2.minEnclosingCircle(best)
    x, y, bw, bh = cv2.boundingRect(best)
    fill_ratio = area / (np.pi * enclosing_r ** 2 + 1e-6)

    cx, cy = float(cx), float(cy)

    if fill_ratio >= 0.65:
        return 'circular', cx, cy, float(enclosing_r), float(enclosing_r), best
    else:
        part_cx = x + bw / 2.0
        part_cy = y + bh / 2.0
        return 'bracket', part_cx, part_cy, float(bw), float(bh), best


# ── Intensity validation at a predicted position ──────────────────────────────
def detect_circular_rotation(gray: np.ndarray,
                              part_cx: float, part_cy: float, R: float,
                              n_holes: int = 3,
                              dist_ratio: float = 0.90) -> list:
    """
    Find the angular positions of the bolt holes on a circular disc by scanning
    the bolt-hole circle (radius = dist_ratio * R) and picking the darkest
    n_holes sectors.

    Returns a sorted list of angles (degrees) where holes were found, or
    an empty list if detection is unreliable.
    """
    h_img, w_img = gray.shape
    sample_r = dist_ratio * R

    # Sample 360 intensity values around the bolt-hole circle
    angles_deg = np.linspace(0, 360, 360, endpoint=False)
    intensities = []
    for a in angles_deg:
        rad = np.radians(a)
        px = int(part_cx + sample_r * np.cos(rad))
        py = int(part_cy + sample_r * np.sin(rad))
        if 0 <= px < w_img and 0 <= py < h_img:
            # Sample a small 5×5 patch to reduce noise
            patch = gray[max(0, py-2):py+3, max(0, px-2):px+3]
            intensities.append(float(np.mean(patch)) if patch.size > 0 else 255.0)
        else:
            intensities.append(255.0)

    # Smooth the intensity profile to merge nearby dark pixels into one dip
    intensities = np.array(intensities)
    kernel = np.ones(9) / 9
    smoothed = np.convolve(intensities, kernel, mode='same')

    # Find the n_holes deepest (darkest) local minima separated by ≥ 80°
    found_angles = []
    remaining = smoothed.copy()
    for _ in range(n_holes):
        idx = int(np.argmin(remaining))
        found_angles.append(float(angles_deg[idx]))
        # Suppress 40° on either side so we don't pick the same hole twice
        for offset in range(-40, 41):
            remaining[(idx + offset) % 360] = 255.0

    found_angles.sort()

    # Sanity check: holes should be roughly 120° apart (±30°)
    if len(found_angles) == 3:
        diffs = [(found_angles[(i+1) % 3] - found_angles[i]) % 360
                 for i in range(3)]
        if all(60 < d < 180 for d in diffs):
            return found_angles   # reliable

    return []  # unreliable, fall back to blueprint angles


def sample_region(gray: np.ndarray, cx: int, cy: int,
                  inner_r: int, outer_r: int):
    """Mean intensity inside inner_r and in annulus inner_r..outer_r."""
    h, w = gray.shape
    bm = np.zeros((h, w), np.uint8)
    rm = np.zeros((h, w), np.uint8)
    if inner_r > 0:
        cv2.circle(bm, (cx, cy), inner_r, 255, -1)
    cv2.circle(rm, (cx, cy), max(inner_r + 2, outer_r), 255, -1)
    cv2.circle(rm, (cx, cy), max(1, inner_r - 2), 0, -1)
    ip = gray[bm > 0]
    rp = gray[rm > 0]
    interior = float(np.mean(ip)) if len(ip) > 0 else 255.0
    collar   = float(np.mean(rp)) if len(rp) > 0 else 128.0
    return interior, collar


# ── Per-image detection ───────────────────────────────────────────────────────
def detect_and_draw(img_path: Path) -> dict:
    orig = cv2.imread(str(img_path))
    if orig is None:
        return {"image": img_path.name, "error": "cannot load"}

    gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    h_img, w_img = gray.shape

    part_type, part_cx, part_cy, dim_w, dim_h, main_c = detect_part(gray)
    if part_type is None:
        return {"image": img_path.name, "error": "part not found",
                "part_type": "unknown"}

    overlay = orig.copy()

    # Draw part boundary
    if part_type == 'circular':
        cv2.circle(overlay, (int(part_cx), int(part_cy)),
                   int(dim_w), (0, 200, 0), 2)
    else:
        x, y, bw, bh = cv2.boundingRect(main_c)
        cv2.rectangle(overlay, (x, y), (x + bw, y + bh), (0, 200, 0), 2)

    results = []

    if part_type == 'circular':
        R = dim_w   # outer radius

        # Try to auto-detect bolt hole positions from the image itself
        detected_angles = detect_circular_rotation(gray, part_cx, part_cy, R)
        use_detected = len(detected_angles) == 3

        for feat_idx, (name, r_ratio, dist_ratio, angle_deg, tol) in enumerate(CIRCULAR_HOLES):
            pred_r = int(r_ratio * R)

            if angle_deg is None:
                # Centre bore — always at the centre
                px, py = int(part_cx), int(part_cy)
            elif use_detected:
                # Use the auto-detected bolt hole angle for this slot
                # Match blueprint hole index to the detected angle that is
                # closest to the original blueprint angle
                bp_rad   = np.radians(angle_deg)
                bp_cos   = np.cos(bp_rad)
                bp_sin   = np.sin(bp_rad)
                best_a   = min(detected_angles,
                               key=lambda a: abs(
                                   np.cos(np.radians(a)) * bp_cos
                                   + np.sin(np.radians(a)) * bp_sin - 1.0))
                rad  = np.radians(best_a)
                px   = int(part_cx + dist_ratio * R * np.cos(rad))
                py   = int(part_cy + dist_ratio * R * np.sin(rad))
                # Remove used angle so the next hole gets a different one
                detected_angles = [a for a in detected_angles if a != best_a]
            else:
                rad = np.radians(angle_deg)
                px  = int(part_cx + dist_ratio * R * np.cos(rad))
                py  = int(part_cy + dist_ratio * R * np.sin(rad))

            px = max(pred_r, min(w_img - pred_r - 1, px))
            py = max(pred_r, min(h_img - pred_r - 1, py))

            inner_r = max(1, int(pred_r * 0.80))
            outer_r = max(inner_r + 4, int(pred_r * 1.60))
            interior, collar = sample_region(gray, px, py, inner_r, outer_r)
            contrast = abs(collar - interior)
            found = contrast > 10 or interior < 70

            results.append({"name": name, "found": found,
                             "px": px, "py": py, "pred_r": pred_r,
                             "interior": round(interior, 1),
                             "contrast": round(contrast, 1)})

            col   = (0, 0, 255) if found else (0, 80, 255)
            thick = 4 if found else 2
            if pred_r > 0:
                cv2.circle(overlay, (px, py), pred_r, col, thick)
                cv2.circle(overlay, (px, py), max(3, pred_r // 3), col, -1)
            status_txt = name.replace("_", " ") + (
                " r=%d" % pred_r if found else " MISSING r=%d" % pred_r)
            cv2.putText(overlay, status_txt,
                        (px + pred_r + 4, py),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)

    else:  # bracket
        W, H = dim_w, dim_h
        for name, rel_x, rel_y, r_ratio, is_circle, tol in BRACKET_FEATURES:
            px = int(part_cx + rel_x * W)
            py = int(part_cy + rel_y * H)
            px = max(5, min(w_img - 5, px))
            py = max(5, min(h_img - 5, py))

            if is_circle:
                pred_r = max(4, int(r_ratio * W))
                inner_r = max(1, int(pred_r * 0.80))
                outer_r = max(inner_r + 4, int(pred_r * 1.60))
                interior, collar = sample_region(gray, px, py, inner_r, outer_r)
                contrast = abs(collar - interior)
                found = contrast > 8 or interior < 80

                col   = (0, 0, 255) if found else (0, 80, 255)
                thick = 4 if found else 2
                cv2.circle(overlay, (px, py), pred_r, col, thick)
                cv2.circle(overlay, (px, py), max(3, pred_r // 3), col, -1)
                cv2.putText(overlay, name.replace("_", " ") + (
                    "" if found else " MISSING"),
                    (px + pred_r + 4, py),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, col, 2)
            else:
                # Square cutout — check for darker rectangular region
                cut_w = int(0.50 * W); cut_h = int(0.30 * H)
                x1 = max(0, px - cut_w // 2); x2 = min(w_img, px + cut_w // 2)
                y1 = max(0, py - cut_h // 2); y2 = min(h_img, py + cut_h // 2)
                roi = gray[y1:y2, x1:x2]
                interior = float(np.mean(roi)) if roi.size > 0 else 255.0
                # Surrounding annulus
                surr = gray[max(0, y1-15):min(h_img, y2+15),
                            max(0, x1-15):min(w_img, x2+15)]
                collar = float(np.mean(surr)) if surr.size > 0 else 128.0
                contrast = abs(collar - interior)
                found = contrast > 8 or interior < 100

                col = (0, 0, 255) if found else (0, 80, 255)
                thick = 4 if found else 2
                cv2.rectangle(overlay, (x1, y1), (x2, y2), col, thick)
                cv2.putText(overlay, "square cutout" + (
                    "" if found else " MISSING"),
                    (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)

                interior_val = round(interior, 1)
                contrast_val = round(contrast, 1)
                found_val    = found

            results.append({"name": name, "found": found,
                             "px": px, "py": py,
                             "pred_r": int(r_ratio * W) if is_circle else 0,
                             "interior": round(interior, 1),
                             "contrast": round(contrast, 1)})

    n_found = sum(1 for r in results if r["found"])
    n_total = len(results)
    status  = "ALL_FOUND" if n_found == n_total else f"MISSING_{n_total - n_found}"

    # Summary bar
    bar = f"{part_type.upper()}: {n_found}/{n_total}  {status}"
    cv2.putText(overlay, bar, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)

    out_path = OUTPUT_DIR / (img_path.stem + "_bp_guided.jpg")
    cv2.imwrite(str(out_path), overlay, [cv2.IMWRITE_JPEG_QUALITY, 90])

    return {"image":     img_path.name,
            "part_type": part_type,
            "n_found":   n_found,
            "n_total":   n_total,
            "status":    status,
            "overlay":   str(out_path),
            "features":  results}


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    images = sorted(p for p in INPUT_DIR.iterdir()
                    if p.suffix.lower() in IMAGE_EXTS)
    print(f"Processing {len(images)} images  ->  {OUTPUT_DIR}")
    print("=" * 72)

    all_results = []
    t0 = time.time()
    counters = {"circular": {"all": 0, "partial": 0},
                "bracket":  {"all": 0, "partial": 0},
                "unknown":  0}

    for i, img_path in enumerate(images, 1):
        r = detect_and_draw(img_path)
        all_results.append(r)
        pt   = r.get("part_type", "unknown")
        nf   = r.get("n_found", 0)
        nt   = r.get("n_total", 4)
        stat = r.get("status", r.get("error", "?"))
        print(f"  [{i:3d}/{len(images)}] {img_path.name[:50]:<50}  "
              f"{pt:<10}  {nf}/{nt}  {stat}")
        if pt in counters:
            if stat == "ALL_FOUND":
                counters[pt]["all"] += 1
            else:
                counters[pt]["partial"] += 1
        else:
            counters["unknown"] += 1

    elapsed = time.time() - t0
    print()
    print("=" * 72)
    print(f"SUMMARY  ({len(images)} images  {elapsed:.1f}s)")
    for pt in ("circular", "bracket"):
        a = counters[pt]["all"]
        p = counters[pt]["partial"]
        print(f"  {pt:<10}  all_found={a}  partial={p}  total={a+p}")
    print(f"  unknown    {counters['unknown']}")
    print(f"\nOverlays:  {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
