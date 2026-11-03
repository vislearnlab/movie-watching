import argparse
import json
import math
import os

from PIL import Image

from structured_caption import structured_caption_image
from caption_frame import load_model

# --- Reference values from the real Henderson & Hayes coarse grid ---
REF_IMG_W, REF_IMG_H = 1024, 768
REF_PATCH_DIAMETER = 205   # px, "coarse" scale
REF_PATCH_DENSITY = 108    # patches per scene, "coarse" scale
# ----------------------------------------------------------------------


def compute_grid(img_w: int, img_h: int, density: int = REF_PATCH_DENSITY,
                  ref_diameter: int = REF_PATCH_DIAMETER,
                  ref_w: int = REF_IMG_W, ref_h: int = REF_IMG_H):
    """Compute tile center points and patch diameter for an image of size
    (img_w, img_h), following the same method as create_scene_patches.m.
    """
    # Patch center spacing -- same formula as the MATLAB code. This
    # naturally adapts to whatever image size is passed in, since it's
    # normalized by the image area.
    px_freq = round(math.sqrt(img_h * img_w) / math.sqrt(density))

    # Scale patch diameter proportionally so tiles keep the same relative
    # size to the frame, even if frame resolution != the original 1024x768.
    scale = math.sqrt((img_h * img_w) / (ref_h * ref_w))
    patch_diameter = round(ref_diameter * scale)

    # Build a grid of center points (mirrors MATLAB's meshgrid + offset)
    xs = list(range(px_freq, img_w + 1, px_freq))
    ys = list(range(px_freq, img_h + 1, px_freq))

    x_offset = px_freq - ((img_w - max(xs)) + px_freq) / 2
    y_offset = px_freq - ((img_h - max(ys)) + px_freq) / 2

    centers = [(x - x_offset, y - y_offset) for y in ys for x in xs]
    return centers, patch_diameter


def crop_tile(image: Image.Image, center, diameter: int):
    """Square bounding-box crop around a tile center.
    NOTE: the original method uses a circular mask, not a square crop --
    this is a simplification. Swap in a PIL ImageDraw circular mask here
    if exact parity with the original stimuli is needed."""
    cx, cy = center
    r = diameter / 2
    left = max(int(cx - r), 0)
    upper = max(int(cy - r), 0)
    right = min(int(cx + r), image.width)
    lower = min(int(cy + r), image.height)
    return image.crop((left, upper, right, lower)), (left, upper, right, lower)


def process_frame(frame_path: str, outdir: str, density: int = REF_PATCH_DENSITY):
    os.makedirs(outdir, exist_ok=True)
    image = Image.open(frame_path).convert("RGB")

    centers, patch_diameter = compute_grid(image.width, image.height, density)
    print(f"Frame size: {image.width}x{image.height} | "
          f"patch diameter: {patch_diameter}px | {len(centers)} tiles")

    llm = load_model()

    results = []
    for i, center in enumerate(centers):
        tile_img, bbox = crop_tile(image, center, patch_diameter)
        if tile_img.width == 0 or tile_img.height == 0:
            continue

        try:
            caption = structured_caption_image(llm, tile_img)
        except ValueError as e:
            print(f"[warn] tile {i} caption failed: {e}")
            caption = None

        results.append({
            "tile_index": i,
            "center_px": center,
            "pixel_bbox": bbox,
            "caption": caption,
        })
        print(f"  tile {i}/{len(centers)}: {caption}")

    frame_name = os.path.splitext(os.path.basename(frame_path))[0]
    out_path = os.path.join(outdir, f"{frame_name}_gaze_tiles.json")
    with open(out_path, "w") as f:
        json.dump({
            "frame_path": frame_path,
            "patch_diameter_px": patch_diameter,
            "num_tiles": len(results),
            "tiles": results,
        }, f, indent=2)

    print(f"Wrote {len(results)} tile captions to {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame", required=True)
    parser.add_argument("--outdir", default="gaze_tile_output")
    parser.add_argument("--density", type=int, default=REF_PATCH_DENSITY)
    args = parser.parse_args()

    process_frame(args.frame, args.outdir, args.density)


if __name__ == "__main__":
    main()
