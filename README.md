# Sketch2Real
#### Generating realistic images from a colored sketch using a diffusion model based on a conditional U-Net
#
## Getting the Dataset
 I used the coco_dataset for this project. I generate the sketch, do the following steps:
#
 1. Convert it to Grey Scale
 2. Invert the grey scale
 3. Apply Gaussian Blur
 4. Invert the pixels
 5. Get the edges by binary_thresholding
 6. Replace the egdes with the original colors

## Model Architechture:
### I tested 2 models
### Model - 1
In this model, I used 128 x 128 size for the image and had 5k images in my dataset
To add noise to the images I used cosine diffusion schedule
The U-Net Architcture is as follows:  
#### INPUT
 (noisy image + sketch)
         ↓
 Initial Conv (6 → 64)
         ↓
 + timestep embedding added
         ↓
 ────────────────────────────
#### ENCODER (Downsampling)
────────────────────────────
 DownBlock 1:  → 32 channels
 DownBlock 2:  → 64 channels
 DownBlock 3:  → 128 channels
          ↓
────────────────────────────
#### (Bottleneck)
────────────────────────────
 ResidualBlock → 256
 ResidualBlock → 256
 ResidualBlock → 128
         ↓
────────────────────────────
#### DECODER (Upsampling)
────────────────────────────
 UpBlock 1:  → 64 channels  (+ skip)
 UpBlock 2:  → 32 channels  (+ skip)
 UpBlock 3:  → 16 channels  (+ skip)
        ↓
#### Final Conv (16 → 3)
         ↓
#### OUTPUT
 (predicted noise image)

