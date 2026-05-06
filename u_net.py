from PIL import Image
import torchvision.transforms as T
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
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

# # load image as tensor — no normalization
# image_path = "coco_dataset/images/val2017/000000091619.jpg"
# img    = Image.open(image_path).convert("RGB")
# tensor = T.ToTensor()(img).unsqueeze(0)              # (1, 3, H, W), range [0,1]

# # pick noise level
# t = torch.tensor([[[[0.5]]]])
# noise_rates, signal_rates = cosine_diffusion_schedule(t)
# noisy_tensor, noise = add_noise(tensor, noise_rates, signal_rates)

# show
# def to_img(x):
#     return x.clamp(0, 1).squeeze().permute(1, 2, 0).numpy()

# fig, axes = plt.subplots(1, 2, figsize=(8, 4))
# axes[0].imshow(to_img(tensor));       axes[0].set_title("Original");     axes[0].axis("off")
# axes[1].imshow(to_img(noisy_tensor)); axes[1].set_title("Noisy (t=0.5)"); axes[1].axis("off")
# plt.tight_layout()
# plt.show()

def sinusoidal_embedding(x, embedding_dim=32):
    frequencies    = torch.exp(torch.linspace(math.log(1.0), math.log(1000.0), embedding_dim // 2, device=x.device))
    angular_speeds = 2.0 * math.pi * frequencies
    embeddings     = torch.cat([torch.sin(angular_speeds * x), torch.cos(angular_speeds * x)], dim=1)
    return embeddings


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # input size and output size may differ.
        # so we need a 1x1 conv to match the dimensions for the residual connection.
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_conv = nn.Identity()
        
        self.batch_norm = nn.BatchNorm2d(in_channels, affine=False) 
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = self.residual_conv(x)
        x = self.batch_norm(x)
        x = F.silu(self.conv1(x))  # using SiLU activation. could have used ELU too
        x = self.conv2(x)
        return x + residual


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, block_depth):
        super().__init__()
        self.block_depth = block_depth
        self.res_blocks = nn.ModuleList()
        for i in range(block_depth):
            if i == 0: # 1st block takes in_channels, rest take out_channels
                self.res_blocks.append(ResidualBlock(in_channels, out_channels))
            else:
                self.res_blocks.append(ResidualBlock(out_channels, out_channels))
    
    def forward(self, x, skips):
        for block in self.res_blocks:
            x = block(x)
            skips.append(x)
        x = F.avg_pool2d(x, kernel_size=2)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, block_depth, skip_channels):
        super().__init__()
        self.block_depth = block_depth
        self.res_blocks = nn.ModuleList()
        for i in range(block_depth):
            if i == 0: # 1st block takes in_channels, rest take out_channels
                in_ch = in_channels + skip_channels
            else:
                in_ch = out_channels + skip_channels
            self.res_blocks.append(ResidualBlock(in_ch, out_channels))
    
    def forward(self, x, skips):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        for block in self.res_blocks:
            skip = skips.pop() # get the corresponding skip connection
            # concatenate skip connection with current feature map
            x = torch.cat([x, skip], dim=1)
            x = block(x)
        return x
    

