import cv2
import numpy as np
from pathlib import Path
import sys   

"""
Convert a photo to a colored sketch drawing.
Edge pixels take the color of the corresponding pixel in the original image.
Everything else is white
Args:
    image_path:     path to input photo
    output_path:    path to save the colored edge drawing
    blur_radius:    gaussian blur kernel size for dodge sketch (must be odd)
    line_thickness: dilation kernel size to thicken edges (1 = no thickening)
Returns:
    colored edge image as numpy array (H, W, 3) BGR
"""

def photo_to_colored_edges(image_path: str, output_path: str,
                            blur_radius: int = 21,
                            line_thickness: int = 2) -> np.ndarray:
    
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inv_gray = 255 - gray
    blurred = cv2.GaussianBlur(inv_gray, (blur_radius, blur_radius), 0)
    inv_blur = 255 - blurred
    sketch = cv2.divide(gray, inv_blur, scale=255.0)

    # Threshold to get binary edges
    _, edges = cv2.threshold(sketch, 220, 255, cv2.THRESH_BINARY_INV)

    if line_thickness > 1:
        if line_thickness % 2 == 0:
            line_thickness += 1  # odd kernel required for size for dilation
        kernel = np.ones((line_thickness, line_thickness), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)

    result = np.ones_like(img) * 255
    # Set edge pixels to original color
    result[edges > 0] = img[edges > 0]

    # Save
    cv2.imwrite(output_path, result)
    return result

if __name__ == "__main__":   

    root = Path(__file__).parent.parent
    val_dir = root / "Sketch2Real" / "coco_dataset" / "images" / "val2017"
    sketch_dir = root / "Sketch2Real" / "sketch"

    input_path = Path(val_dir)
    output_path = Path(sketch_dir)

    # create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)

    count = 0

    image_files = []
    for f in input_path.rglob("*.jpg"):
        image_files.append(f)

    for img_file in image_files:
        relative = img_file.relative_to(input_path)
        out_file = output_path / relative
        out_file.parent.mkdir(parents=True, exist_ok=True)        

        try:
            photo_to_colored_edges(str(img_file), str(out_file))            
            if count % 500 == 0:
                print(f"{count}/5000 images done...")
        except Exception as e:
            print(f"FAILED {img_file.name}: {e}")
        
        count += 1

    print("Done")
