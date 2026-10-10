"""
Blueprint-Matched Hole Detection
=================================
For every image in data/inputs/:

  Step 1 — Detect part boundary in the photo (find outer circle or bounding rect).
  Step 2 — Detect ALL holes in the photo using dark-region analysis,
            completely independent of blueprint knowledge.
  Step 3 — Identify which blueprint matches this part (circular or bracket).
  Step 4 — Detect ALL holes in the matching blueprint using the same
            dark-region method on the blueprint image.
  Step 5 — Convert both sets of holes to scale-invariant coordinates
            (position as fraction of part radius/size, radius as fraction).
  Step 6 — Match photo holes to blueprint holes using nearest-neighbour
            on scale-invariant position. Tolerance = 0.12 * R.
  Step 7 — Draw output overlay:
              GREEN  = hole found in photo AND matches blueprint
              RED    = blueprint hole NOT found in photo (missing)
              YELLOW = hole found in photo but not in blueprint (extra)

Usage:
    python tools/match_holes_to_blueprint.py
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np

INPUT_DIR  = Path("data/inputs")
BP_DIR     = Path("data/blueprints")
OUTPUT_DIR = Path("outputs/hole_match")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# ── Matching tolerance (fraction of part radius) ─────────────────────────────
MATCH_TOLERANCE = 0.12


# ─────────────────────────────────────────────────────────────────────────────
# Part detection
# ─────────────────────────────────────────────────────────────────────────────

def find_part(gray: np.ndarray):
    """
    Returns (part_type, cx, cy, R, bbox_w, bbox_h, contour).
    part_type: 'circular' | 'bracket'
    R = outer radius for circular, bbox_w for bracket (used for normalisation).
    """
    _, binary = cv2.threshold(gray, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark = cv2.bitwise_not(binary)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, k, iterations=3)
    contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, None, None, None, None, None

    H, W = gray.shape
    BORDER = 25
    best, best_score = None, -1.0
    for c in contours:
        area = cv2.contourArea(c)
        if area < 1500:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        if (x <= BORDER or y <= BORDER
                or x + bw >= W - BORDER or y + bh >= H - BORDER):
            continue
        (cx, cy), r = cv2.minEnclosingCircle(c)
        fill = area / (np.pi * r * r + 1e-6)
        if area * fill > best_score:
            best_score = area * fill
            best = c

    if best is None:
        best = max(contours, key=cv2.contourArea)

    area = cv2.contourArea(best)
    if area < 1500:
        return None, None, None, None, None, None, None

    (cx, cy), enc_r = cv2.minEnclosingCircle(best)
    x, y, bw, bh = cv2.boundingRect(best)
    fill = area / (np.pi * enc_r * enc_r + 1e-6)

    if fill >= 0.65:
        return 'circular', float(cx), float(cy), float(enc_r), float(bw), float(bh), best
    else:
        return 'bracket', x + bw / 2.0, y + bh / 2.0, float(bw), float(bw), float(bh), best


# ─────────────────────────────────────────────────────────────────────────────
# Hole detection  (works on both photos and blueprints)
# ─────────────────────────────────────────────────────────────────────────────

def detect_holes_in_image(gray: np.ndarray,
                           part_cx: float, part_cy: float, part_R: float,
                           min_r_ratio: float = 0.010,
                           max_r_ratio: float = 0.35,
                           is_blueprint: bool = False) -> list:
    """
    Detect all circular holes inside the part boundary.

    For product photos:  look for dark filled blobs (through-holes are dark).
    For blueprints:      look for small circular ring outlines (drawn as rings).

    Returns list of dicts:
        cx, cy  — centre in image pixels
        r       — radius in pixels
        dist_r  — distance from part centre / part_R  (scale-invariant)
        r_ratio — radius / part_R                     (scale-invariant)
        angle   — angle from part centre (degrees)
    """
    h, w = gray.shape
    mask_area = np.zeros((h, w), np.uint8)
    cv2.circle(mask_area, (int(part_cx), int(part_cy)), int(part_R * 1.05), 255, -1)

    min_r_px = max(3, int(min_r_ratio * part_R))
    max_r_px = int(max_r_ratio * part_R)

    holes = []

    if is_blueprint:
        # Blueprints: thin dark ring outlines on white background.
        # Use Hough on lightly blurred grayscale to find the ring circles.
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        # Mask to part interior
        bp_masked = np.where(mask_area > 0, blurred, 255).astype(np.uint8)
        raw = cv2.HoughCircles(
            bp_masked, cv2.HOUGH_GRADIENT,
            dp=1, minDist=int(min_r_px * 1.5),
            param1=60, param2=12,
            minRadius=min_r_px, maxRadius=max_r_px,
        )
        if raw is not None:
            for cx, cy, r in np.round(raw[0, :]).astype(float):
                # Validate: must be inside part
                dist = np.sqrt((cx - part_cx)**2 + (cy - part_cy)**2)
                if dist + r > part_R * 1.05:
                    continue
                # Validate: the circle boundary should be dark (ring outline)
                ring_mask = np.zeros((h, w), np.uint8)
                cv2.circle(ring_mask, (int(cx), int(cy)), int(r), 255, max(1, int(r * 0.4)))
                ring_px = gray[ring_mask > 0]
                if len(ring_px) == 0 or np.mean(ring_px) > 220:
                    continue
                dist_r = dist / part_R
                holes.append({
                    "cx": cx, "cy": cy, "r": r,
                    "dist_r": dist_r,
                    "r_ratio": r / part_R,
                    "angle": float(np.degrees(np.arctan2(cy - part_cy, cx - part_cx))),
                })
    else:
        # Product photos: dark filled blobs = through-holes.
        # Use multi-threshold dark-region detection (from circle_extractor).
        product_pixels = gray[mask_area > 0]
        if len(product_pixels) == 0:
            return holes

        # Adaptive thresholds (low range for very dark holes)
        base_thr = int(np.percentile(product_pixels, 15) + 5)
        base_thr = min(base_thr, 55)
        scan_thresholds = sorted(set([15, 20, 25, 35, base_thr]))

        k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        prod_uint8 = (mask_area > 0).astype(np.uint8) * 255

        blob_candidates = []
        DEDUP = 10.0

        for thr in scan_thresholds:
            dark = ((gray < thr) & (mask_area > 0)).astype(np.uint8) * 255
            dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, k3, iterations=1)
            contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                area = cv2.contourArea(c)
                if area < np.pi * min_r_px ** 2:
                    continue
                p = cv2.arcLength(c, True)
                if p == 0:
                    continue
                circ = 4 * np.pi * area / (p * p)
                if circ < 0.40:
                    continue
                hull = cv2.convexHull(c)
                ha = cv2.contourArea(hull)
                sol = area / ha if ha > 0 else 0.0
                if sol < 0.60:
                    continue
                (cx, cy), r = cv2.minEnclosingCircle(c)
                if r < min_r_px or r > max_r_px:
                    continue
                # Mask overlap
                hm = np.zeros((h, w), np.uint8)
                cv2.drawContours(hm, [c], -1, 255, -1)
                inter = np.count_nonzero(cv2.bitwise_and(hm, prod_uint8))
                if area > 0 and inter / area < 0.75:
                    continue
                # Interior darkness
                ip = gray[hm > 0]
                if len(ip) == 0:
                    continue
                df = float(np.mean(ip < thr))
                if df < 0.75:
                    continue
                blob_candidates.append((cx, cy, r, circ, sol, df, c, thr))

        # Deduplicate by centre proximity, keep highest circularity
        blob_candidates.sort(key=lambda x: -x[3])
        accepted_centres = []
        for bc in blob_candidates:
            cx, cy = bc[0], bc[1]
            if any(((cx - ax)**2 + (cy - ay)**2)**0.5 < DEDUP
                   for ax, ay in accepted_centres):
                continue
            accepted_centres.append((cx, cy))
            r = bc[2]
            dist = np.sqrt((cx - part_cx)**2 + (cy - part_cy)**2)
            holes.append({
                "cx": float(cx), "cy": float(cy), "r": float(r),
                "dist_r": dist / part_R,
                "r_ratio": r / part_R,
                "angle": float(np.degrees(np.arctan2(cy - part_cy, cx - part_cx))),
            })

        # Also run large-bore detection (CLAHE Hough for centre bore / large rings)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enh   = clahe.apply(gray)
        bl    = cv2.GaussianBlur(
            np.where(mask_area > 0, enh, 0).astype(np.uint8), (5, 5), 0)
        raw = cv2.HoughCircles(
            bl, cv2.HOUGH_GRADIENT,
            dp=1, minDist=60,
            param1=80, param2=15,
            minRadius=int(max_r_ratio * part_R * 0.5),
            maxRadius=int(max_r_ratio * part_R),
        )
        if raw is not None:
            for cx, cy, r in np.round(raw[0, :]).astype(float):
                # Interior-vs-collar contrast
                bm = np.zeros((h, w), np.uint8)
                cv2.circle(bm, (int(cx), int(cy)), max(1, int(r - 6)), 255, -1)
                cm = np.zeros((h, w), np.uint8)
                cv2.circle(cm, (int(cx), int(cy)), int(r + 20), 255, -1)
                cv2.circle(cm, (int(cx), int(cy)), int(r + 4), 0, -1)
                cm = cv2.bitwise_and(cm, prod_uint8)
                ip = gray[bm > 0]
                cp = gray[cm > 0]
                if len(ip) < 30 or len(cp) < 30:
                    continue
                contrast = abs(float(np.mean(cp)) - float(np.mean(ip)))
                if contrast < 15:
                    continue
                # Mask overlap
                if np.count_nonzero(bm) > 0:
                    ov = np.count_nonzero(cv2.bitwise_and(bm, prod_uint8))
                    if ov / np.count_nonzero(bm) < 0.80:
                        continue
                # No duplicate
                if any(((cx - ax)**2 + (cy - ay)**2)**0.5 < r * 0.5
                       for ax, ay in accepted_centres):
                    continue
                dist = np.sqrt((cx - part_cx)**2 + (cy - part_cy)**2)
                holes.append({
                    "cx": float(cx), "cy": float(cy), "r": float(r),
                    "dist_r": dist / part_R,
                    "r_ratio": r / part_R,
                    "angle": float(np.degrees(np.arctan2(
                        cy - part_cy, cx - part_cx))),
                })
                accepted_centres.append((float(cx), float(cy)))

    return holes


# ─────────────────────────────────────────────────────────────────────────────
# Blueprint selection and hole extraction
# ─────────────────────────────────────────────────────────────────────────────

def get_blueprint_holes(part_type: str) -> tuple:
    """
    Load the matching blueprint, find the part, detect all holes,
    return (bp_holes_normalised, bp_image, bp_part_cx, bp_part_cy, bp_R).
    """
    bp_name = "circular_rear.png" if part_type == "circular" else "box_front.png"
    bp_path = BP_DIR / bp_name
    bp      = cv2.imread(str(bp_path))
    bp_gray = cv2.cvtColor(bp, cv2.COLOR_BGR2GRAY)

    # Find part in blueprint
    bp_type, bp_cx, bp_cy, bp_R, bp_bw, bp_bh, _ = find_part(bp_gray)

    # If find_part fails on the blueprint (it's mostly white), use the
    # full-image enclosing circle of all dark pixels
    if bp_cx is None:
        dark = (bp_gray < 200).astype(np.uint8) * 255
        nz = np.argwhere(dark > 0)
        if len(nz) == 0:
            return [], bp, 0, 0, 1
        y1, x1 = nz.min(axis=0)
        y2, x2 = nz.max(axis=0)
        bp_cx = float((x1 + x2) / 2)
        bp_cy = float((y1 + y2) / 2)
        bp_R  = float(max(x2 - x1, y2 - y1) / 2)

    bp_holes = detect_holes_in_image(bp_gray, bp_cx, bp_cy, bp_R,
                                      is_blueprint=True)
    return bp_holes, bp, float(bp_cx), float(bp_cy), float(bp_R)


# ─────────────────────────────────────────────────────────────────────────────
# Matching
# ─────────────────────────────────────────────────────────────────────────────

def match_holes(photo_holes: list, bp_holes: list,
                tol: float = MATCH_TOLERANCE) -> tuple:
    """
    Match photo holes to blueprint holes using scale-invariant position.

    Returns:
        matched   : list of (photo_hole, bp_hole) pairs
        missing   : blueprint holes with no photo match
        extra     : photo holes with no blueprint match
    """
    used_photo = set()
    used_bp    = set()
    matched    = []

    # Greedy nearest-neighbour on (dist_r, angle) space
    # Convert angle to a 2-D unit vector to handle wraparound
    def pos_vec(h):
        rad = np.radians(h["angle"])
        return np.array([h["dist_r"] * np.cos(rad),
                         h["dist_r"] * np.sin(rad)])

    # Sort bp holes largest-first so bigger holes get matched first
    bp_order = sorted(range(len(bp_holes)), key=lambda i: -bp_holes[i]["r_ratio"])

    for bi in bp_order:
        bh  = bp_holes[bi]
        bv  = pos_vec(bh)
        best_dist, best_pi = 1e9, -1
        for pi, ph in enumerate(photo_holes):
            if pi in used_photo:
                continue
            pv   = pos_vec(ph)
            dist = float(np.linalg.norm(pv - bv))
            if dist < best_dist:
                best_dist = dist
                best_pi   = pi
        if best_pi >= 0 and best_dist < tol:
            matched.append((photo_holes[best_pi], bh))
            used_photo.add(best_pi)
            used_bp.add(bi)

    missing = [bp_holes[i]  for i in range(len(bp_holes))  if i not in used_bp]
    extra   = [photo_holes[i] for i in range(len(photo_holes)) if i not in used_photo]
    return matched, missing, extra


# ─────────────────────────────────────────────────────────────────────────────
# Draw overlay
# ─────────────────────────────────────────────────────────────────────────────

def draw_overlay(orig: np.ndarray,
                 part_type: str,
                 part_cx: float, part_cy: float, part_R: float,
                 matched: list, missing: list, extra: list,
                 scale: float = 1.0, roi_x: int = 0, roi_y: int = 0
                 ) -> np.ndarray:
    ov = orig.copy()

    def to_orig(cx, cy, r):
        ox = int((cx + roi_x) / scale)
        oy = int((cy + roi_y) / scale)
        orr = max(1, int(r / scale))
        return ox, oy, orr

    # Part boundary
    ocx, ocy, oR = to_orig(part_cx, part_cy, part_R)
    cv2.circle(ov, (ocx, ocy), oR, (0, 200, 0), 2)

    # GREEN — matched (hole found, matches blueprint)
    for ph, bh in matched:
        ox, oy, orr = to_orig(ph["cx"], ph["cy"], ph["r"])
        cv2.circle(ov, (ox, oy), orr, (0, 230, 0), 3)
        cv2.circle(ov, (ox, oy), max(2, orr // 3), (0, 230, 0), -1)

    # RED — missing (in blueprint, not found in photo)
    for bh in missing:
        # Project blueprint hole into photo space using scale-invariant coords
        rad = np.radians(bh["angle"])
        ph_cx = part_cx + bh["dist_r"] * part_R * np.cos(rad)
        ph_cy = part_cy + bh["dist_r"] * part_R * np.sin(rad)
        ph_r  = bh["r_ratio"] * part_R
        ox, oy, orr = to_orig(ph_cx, ph_cy, ph_r)
        cv2.circle(ov, (ox, oy), max(3, orr), (0, 0, 220), 3)
        cv2.putText(ov, "MISSING", (ox + max(3, orr) + 3, oy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 220), 1)

    # YELLOW — extra (found in photo, not in blueprint)
    for ph in extra:
        ox, oy, orr = to_orig(ph["cx"], ph["cy"], ph["r"])
        cv2.circle(ov, (ox, oy), max(3, orr), (0, 210, 210), 2)

    # Legend
    n_m   = len(matched)
    n_mis = len(missing)
    n_ex  = len(extra)
    cv2.putText(ov,
                f"MATCHED={n_m}  MISSING={n_mis}  EXTRA={n_ex}",
                (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
    cv2.putText(ov, "GREEN=found  RED=missing  YELLOW=extra",
                (10, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    return ov


# ─────────────────────────────────────────────────────────────────────────────
# Per-image pipeline
# ─────────────────────────────────────────────────────────────────────────────

# Cache blueprint holes so we don't re-parse them for every image
_bp_cache = {}

def process_image(img_path: Path) -> dict:
    orig = cv2.imread(str(img_path))
    if orig is None:
        return {"image": img_path.name, "error": "cannot load"}

    gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)

    # 1. Find part in photo
    part_type, part_cx, part_cy, part_R, bw, bh, main_c = find_part(gray)
    if part_type is None:
        return {"image": img_path.name, "error": "part not found"}

    # 2. Detect holes in photo
    photo_holes = detect_holes_in_image(gray, part_cx, part_cy, part_R)

    # 3. Get blueprint holes (cached)
    if part_type not in _bp_cache:
        _bp_cache[part_type] = get_blueprint_holes(part_type)
    bp_holes, bp_img, bp_cx, bp_cy, bp_R = _bp_cache[part_type]

    # 4. Match
    matched, missing, extra = match_holes(photo_holes, bp_holes)

    # 5. Draw overlay
    overlay = draw_overlay(orig, part_type,
                            part_cx, part_cy, part_R,
                            matched, missing, extra)

    # Save
    out_path = OUTPUT_DIR / (img_path.stem + "_matched.jpg")
    cv2.imwrite(str(out_path), overlay, [cv2.IMWRITE_JPEG_QUALITY, 92])

    return {
        "image":      img_path.name,
        "part_type":  part_type,
        "photo_holes":  len(photo_holes),
        "bp_holes":     len(bp_holes),
        "matched":      len(matched),
        "missing":      len(missing),
        "extra":        len(extra),
        "status":       "ALL_MATCHED" if len(missing) == 0 else f"MISSING_{len(missing)}",
        "overlay":      str(out_path),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    images = sorted(p for p in INPUT_DIR.iterdir()
                    if p.suffix.lower() in IMAGE_EXTS)
    print(f"Processing {len(images)} images -> {OUTPUT_DIR}")
    print("=" * 72)

    all_results = []
    t0 = time.time()

    for i, img_path in enumerate(images, 1):
        r = process_image(img_path)
        all_results.append(r)
        if "error" in r:
            print(f"  [{i:3d}/{len(images)}] {img_path.name[:50]:<50}  ERROR: {r['error']}")
        else:
            print(f"  [{i:3d}/{len(images)}] {img_path.name[:50]:<50}  "
                  f"{r['part_type']:<10}  "
                  f"photo={r['photo_holes']:2d}  bp={r['bp_holes']:2d}  "
                  f"matched={r['matched']:2d}  missing={r['missing']:2d}  "
                  f"extra={r['extra']:2d}  {r['status']}")

    elapsed = time.time() - t0
    ok   = [r for r in all_results if r.get("status") == "ALL_MATCHED"]
    mis  = [r for r in all_results if "MISSING" in r.get("status", "")]
    err  = [r for r in all_results if "error" in r]

    print()
    print("=" * 72)
    print(f"SUMMARY  ({len(images)} images  {elapsed:.1f}s)")
    print(f"  All holes matched  : {len(ok)}")
    print(f"  Some missing       : {len(mis)}")
    print(f"  Errors             : {len(err)}")
    print(f"Overlays saved to    : {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
