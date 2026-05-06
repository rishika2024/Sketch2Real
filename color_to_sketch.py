import cv2
import numpy as np
from pathlib import Path


def photo_to_colored_edges(image_path: str, output_path: str,
                            blur_radius: int = 21,
                            line_thickness: int = 2) -> np.ndarray:
    """
    Convert a photo to a colored edge drawing.
    Edge pixels take the color of the corresponding pixel in the original image.
    Everything else is white (like a blank canvas).

    Args:
        image_path:     path to input photo
        output_path:    path to save the colored edge drawing
        blur_radius:    gaussian blur kernel size for dodge sketch (must be odd)
        line_thickness: dilation kernel size to thicken edges (1 = no thickening)

    Returns:
        colored edge image as numpy array (H, W, 3) BGR
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inv_gray = 255 - gray
    blurred  = cv2.GaussianBlur(inv_gray, (blur_radius, blur_radius), 0)
    inv_blur = 255 - blurred
    sketch   = cv2.divide(gray, inv_blur, scale=255.0)  # classic dodge pencil sketch

    # sketch: dark strokes on white — threshold to binary stroke mask
    _, edges = cv2.threshold(sketch, 220, 255, cv2.THRESH_BINARY_INV)

    if line_thickness > 1:
        kernel = np.ones((line_thickness, line_thickness), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)

    result = np.ones_like(img) * 255
    result[edges > 0] = img[edges > 0]

    # Save
    cv2.imwrite(output_path, result)
    return result


def process_dataset(input_dir: str, output_dir: str):
    """
    Process an entire folder of images into colored edge drawings.
    Output folder structure mirrors input folder structure.
    Paired images share the same filename — easy to load for training.

    Args:
        input_dir:   folder of real photos  (your 'labels' / ground truth)
        output_dir:  folder to save colored edge drawings  (your 'inputs')
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    image_files = [f for f in input_path.rglob("*") if f.suffix.lower() in extensions]

    if not image_files:
        print(f"No images found in {input_dir}")
        return

    print(f"Processing {len(image_files)} images...")

    success, failed = 0, 0
    for img_file in image_files:
        # Mirror subfolder structure in output
        relative = img_file.relative_to(input_path)
        out_file = output_path / relative
        out_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            photo_to_colored_edges(str(img_file), str(out_file))
            success += 1
            if success % 100 == 0:
                print(f"  {success}/{len(image_files)} done...")
        except Exception as e:
            print(f"  FAILED {img_file.name}: {e}")
            failed += 1

    print(f"\nDone. {success} succeeded, {failed} failed.")
    print(f"Colored edge drawings saved to: {output_dir}")
    print(f"\nDataset pairs:")
    print(f"  Input  (colored edges) → {output_dir}/")
    print(f"  Output (real photos)   → {input_dir}/")


if __name__ == "__main__":
    import sys
    from pathlib import Path

    root = Path(__file__).parent.parent  # sketch2real/
    val_dir = root / "Sketch2Real" / "coco_dataset" / "images" / "val2017"
    sketch_dir = root / "Sketch2Real" / "sketch"

    if not val_dir.exists():
        print(f"ERROR: val2017 not found at {val_dir}")
        print("Download it first: wget http://images.cocodataset.org/zips/val2017.zip")
        sys.exit(1)

    process_dataset(str(val_dir), str(sketch_dir))


# img = cv2.imread('coco_dataset/images/val2017/000000091619.jpg')
# print('Shape (H, W, C):', img.shape)
# print('Size: {}x{}'.format(img.shape[1], img.shape[0]))

# result = photo_to_colored_edges(
#     image_path="coco_dataset/images/val2017/000000000632.jpg",
#     output_path="test_sketch.jpg"
# )
# print("Saved to test_sketch.jpg")


