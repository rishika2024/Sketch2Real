import cv2
import numpy as np
from pathlib import Path
import sys   



def photo_to_colored_edges(train_path, dataset_path, output_path, size):    
    """
    Convert a photo to a colored sketch drawing.
    Edge pixels take the color of the corresponding pixel in the original image.
    Everything else is white
    Args:
        image_path:     path to input photo
        output_path:    path to save the colored edge drawing   
        train_path:     path to original photo to be copied into new dataset (can be same as image_path if you want to overwrite COCO, but better to write into new folder)
        size:           output image size (sketch and photo will be resized to this size after sketching)
    Returns:
        colored edge image as numpy array (H, W, 3) BGR
    """
    # Load image
    img = cv2.imread(train_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {train_path}")


    # ---- CONVERT TO SKETCH FIRST (original resolution) ----
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inv_gray = 255 - gray
    blurred = cv2.GaussianBlur(inv_gray, (19, 19), 0)
    inv_blur = 255 - blurred
    sketch = cv2.divide(gray, inv_blur, scale=255.0)

    # Threshold to get binary edges    
    _, edges = cv2.threshold(sketch, 220, 255, cv2.THRESH_BINARY_INV)    

    result = np.ones_like(img) * 255
    # Set edge pixels to original color
    result[edges > 0] = img[edges > 0]  

    # ---- RESIZE AFTER SKETCH ----
    result = cv2.resize(result, (size, size), interpolation=cv2.INTER_AREA)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)

    # ---- SAVE ----
    cv2.imwrite(output_path, result)
    cv2.imwrite(dataset_path, img)  # write into new dataset

    return result


if __name__ == "__main__":   

    root = Path(__file__).parent.parent
    train_dir = root / "Sketch2Real" / "train2017"
    dataset_dir = root / "Sketch2Real" / "dataset_large"
    sketch_dir = root / "Sketch2Real" / "sketch_large"
    

    train_path = Path(train_dir)
    output_path = Path(sketch_dir)
    dataset_path = Path(dataset_dir)

    # create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    dataset_path.mkdir(parents=True, exist_ok=True)


    count = 0

    image_files = []
    for f in train_path.rglob("*.jpg"):
        image_files.append(f)

    for img_file in image_files:
        relative = img_file.relative_to(train_path)
        out_file = output_path / relative
        ds_file = dataset_path / relative

        out_file.parent.mkdir(parents=True, exist_ok=True)
        ds_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            photo_to_colored_edges(str(img_file), str(ds_file), str(out_file), 256)            
            if count % 500 == 0:
                print(f"{count}/5000 images done...")
        except Exception as e:
            print(f"FAILED {img_file.name}: {e}")
        
        count += 1

    print("Done")
