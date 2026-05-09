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

 {{< figure src="dataset_large/000000204536.jpg" width="60%">}}
  {{< figure src="sketch_large/000000204536.jpg" width="60%">}}

## Model Architechture:
### I tested 2 models
### Model - 1
In this model, I used 128 x 128 size for the image and had 5k images in my dataset
1. **Noise schedule used:** Cosine Noise Schedule
2. **Optimiser:** AdamW
3. **Loss Function:** L1
4. **Activation Function:** SILU

The U-Net Architcture is as follows:  
#### INPUT
 (concatenating noisy image + sketch) -> Initial Conv (6 -> 64) + timestep embedding added (128 channels)      

#### ENCODER (Downsampling)

 DownBlock 1:  128 -> 32 channels
 DownBlock 2:  32 -> 64 channels
 DownBlock 3:  64 -> 128 channels

#### (Bottleneck)

 ResidualBlock1: 128 -> 256
 ResidualBlock2: 256 -> 256
 ResidualBlock3: 256 -> 128

#### DECODER (Upsampling)

 UpBlock 1:  128 -> 64 channels  (+ skip)
 UpBlock 2:  64 -> 32 channels  (+ skip)
 UpBlock 3:  32 -> 16 channels  (+ skip)

#### Final Conv (16 -> 3)

#### OUTPUT
 (predicted noise image)

