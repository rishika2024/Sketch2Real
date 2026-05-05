from PIL import Image
import torchvision.transforms as T
import torch
import math
import matplotlib.pyplot as plt

def cosine_diffusion_schedule(t):
    # t is a tensor of values between 0 and 1
    signal_rates = torch.cos(t * math.pi / 2)
    noise_rates  = torch.sin(t * math.pi / 2)
    return noise_rates, signal_rates

def add_noise(colored_images, noise_rates, signal_rates):
    noise = torch.randn_like(colored_images)
    noisy_images = signal_rates * colored_images + noise_rates * noise
    return noisy_images, noise

# load image as tensor — no normalization
image_path = "coco_dataset/images/val2017/000000091619.jpg"
img    = Image.open(image_path).convert("RGB")
tensor = T.ToTensor()(img).unsqueeze(0)              # (1, 3, H, W), range [0,1]

# pick noise level
t = torch.tensor([[[[0.7]]]])
noise_rates, signal_rates = cosine_diffusion_schedule(t)
noisy_tensor, noise = add_noise(tensor, noise_rates, signal_rates)

# show
def to_img(x):
    return x.clamp(0, 1).squeeze().permute(1, 2, 0).numpy()

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes[0].imshow(to_img(tensor));       axes[0].set_title("Original");     axes[0].axis("off")
axes[1].imshow(to_img(noisy_tensor)); axes[1].set_title("Noisy (t=0.5)"); axes[1].axis("off")
plt.tight_layout()
plt.show()