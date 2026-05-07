import torch
from PIL import Image
import torchvision.transforms as T
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
import copy
from tqdm import tqdm

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


class ConditionalUNet(nn.Module):
    """
    UNet for sketch to photo generation.
    Input: noisy colored image (3 ch) + sketch (3 ch) = 6 channels
    Output: predicted noise (3 channels)
    """
    def __init__(self, image_size=256, noise_embedding_size=64):
        super().__init__()
        self.image_size = image_size
        self.noise_embedding_size = noise_embedding_size

        # initial 1x1 conv: 6 channels (noisy + sketch) -> 64 channels
        self.initial_conv = nn.Conv2d(6, 64, kernel_size=1)

        # after concat with noise embedding
        in_after_concat = 64 + noise_embedding_size

        # encoder: 3 DownBlocks, channels 64 -> 128 -> 256
        self.down1 = DownBlock(in_after_concat, 64,  block_depth=2)
        self.down2 = DownBlock(64,  128, block_depth=2)
        self.down3 = DownBlock(128, 256, block_depth=2)

        # bottleneck: 256 -> 512 -> 512 -> 256
        self.bottleneck1 = ResidualBlock(256, 512)
        self.bottleneck2 = ResidualBlock(512, 512)
        self.bottleneck3 = ResidualBlock(512, 256)

        # decoder:
        self.up1 = UpBlock(256, 128, block_depth=2, skip_channels=256)
        self.up2 = UpBlock(128, 64,  block_depth=2, skip_channels=128)
        self.up3 = UpBlock(64,  32,  block_depth=2, skip_channels=64)

        # final 1x1 conv: 32 -> 3 (predicted noise)
        self.final_conv = nn.Conv2d(32, 3, kernel_size=1)
        nn.init.zeros_(self.final_conv.weight)
        nn.init.zeros_(self.final_conv.bias)


    def forward(self, noisy_images, sketches, noise_variances):

        # concat sketch as condition -> 6 channels
        x = torch.cat([noisy_images, sketches], dim=1)
        x = self.initial_conv(x)

        # sinusoidal embedding of noise variance, broadcast to image size
        noise_emb = sinusoidal_embedding(noise_variances, self.noise_embedding_size)
        noise_emb = noise_emb.view(-1, self.noise_embedding_size, 1, 1)
        noise_emb = F.interpolate(noise_emb, size=(self.image_size, self.image_size), mode="nearest")

        # concat image features with noise embedding
        x = torch.cat([x, noise_emb], dim=1)

        # encoder
        skips = []
        x = self.down1(x, skips)
        x = self.down2(x, skips)
        x = self.down3(x, skips)

        # bottleneck
        x = self.bottleneck1(x)
        x = self.bottleneck2(x)
        x = self.bottleneck3(x)

        # decoder
        x = self.up1(x, skips)
        x = self.up2(x, skips)
        x = self.up3(x, skips)

        # final output: predicted noise
        return self.final_conv(x)

def update_ema(ema_model, model, decay=0.999):
    with torch.no_grad():
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)

class SketchPhotoDataset(Dataset):
    """Loads (sketch, photo) pairs. Filenames must match across both folders."""
    def __init__(self, image_dir, sketch_dir):
        self.image_dir  = Path(image_dir)
        self.sketch_dir = Path(sketch_dir)

        self.filenames = sorted([
            f.name for f in self.image_dir.iterdir()
            if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ])

        self.transform = transforms.ToTensor()  # range [0, 1]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        name   = self.filenames[idx]
        photo  = Image.open(self.image_dir  / name).convert("RGB")
        sketch = Image.open(self.sketch_dir / name).convert("RGB")
        return self.transform(sketch), self.transform(photo)



def train(
    image_dir  = "/content/images",
    sketch_dir = "/content/sketches",
    output_dir      = "/content/drive/MyDrive/gen_ai/checkpoints",
    image_size      = 256,
    batch_size      = 64,
    epochs          = 200,
    lr              = 1e-3,
    weight_decay    = 1e-4,
    ema_decay       = 0.999,
    noise_embedding_size = 64,
    save_every      = 10,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # data
    dataset = SketchPhotoDataset(image_dir, sketch_dir)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                         num_workers=4, pin_memory=True, drop_last=True)
    print(f"Dataset size: {len(dataset)} pairs")

    # model
    model = ConditionalUNet(image_size=image_size,
                            noise_embedding_size=noise_embedding_size).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # EMA copy of the model (used for inference)
    ema_model = copy.deepcopy(model)
    for p in ema_model.parameters():
        p.requires_grad_(False)

    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---------- training loop ----------

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(loader, desc=f"Epoch {epoch}/{epochs}")
        for sketches, photos in pbar:
            sketches = sketches.to(device)
            photos   = photos.to(device)
            B        = photos.shape[0]

            # 1) sample random diffusion times in [0, 1]
            t = torch.rand(B, 1, 1, 1, device=device)
            noise_rates, signal_rates = cosine_diffusion_schedule(t)

            # 2) noise the colored image (NOT the sketch)
            noisy_photos, noise = add_noise(photos, noise_rates, signal_rates)

            # 3) model predicts the noise, conditioned on the sketch
            noise_variances = noise_rates ** 2
            predicted_noise = model(noisy_photos, sketches, noise_variances)

            # 4) MAE loss
            loss = F.l1_loss(predicted_noise, noise)

            # 5) backprop
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # 6) EMA update
            update_ema(ema_model, model, decay=ema_decay)

            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch} avg loss: {avg_loss:.4f}")

        # save checkpoint
        if epoch % save_every == 0 or epoch == epochs:
            ckpt = {
                "epoch":     epoch,
                "model":     model.state_dict(),
                "ema_model": ema_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "avg_loss":  avg_loss,
            }
            torch.save(ckpt, output_dir / f"ckpt_epoch{epoch:04d}.pt")
            print(f"Saved checkpoint at epoch {epoch}")

            # generate a sample to track visual progress
            model.eval()
            with torch.no_grad():
                # grab one sketch from the dataset
                sample_sketch, _ = dataset[0]
                sample_sketch = sample_sketch.unsqueeze(0).to(device)

                # generate a sample to track visual progress
                model.eval()
                with torch.no_grad():
                    sample_sketch, _ = dataset[0]
                    sample_sketch = sample_sketch.unsqueeze(0).to(device)
                    
                    x = torch.randn(1, 3, image_size, image_size, device=device)
                    num_steps = 50
                    
                    # use times that avoid t=1.0 and t=0.0 exactly
                    times = torch.linspace(1.0 - 1e-3, 1e-3, num_steps + 1, device=device)
                    
                    for i in range(num_steps):
                        t      = times[i].view(1, 1, 1, 1)
                        t_next = times[i + 1].view(1, 1, 1, 1)
                        
                        noise_rate       = torch.sin(t * math.pi / 2)
                        signal_rate      = torch.cos(t * math.pi / 2)
                        noise_rate_next  = torch.sin(t_next * math.pi / 2)
                        signal_rate_next = torch.cos(t_next * math.pi / 2)
                        
                        predicted_noise  = model(x, sample_sketch, noise_rate ** 2)
                        predicted_image  = (x - noise_rate * predicted_noise) / signal_rate
                        x = signal_rate_next * predicted_image + noise_rate_next * predicted_noise
                    
                    from torchvision.utils import save_image
                    combined = torch.cat([sample_sketch, predicted_image.clamp(0, 1)], dim=0)
                    save_image(combined, output_dir / f"sample_epoch{epoch:04d}.png", nrow=2)
                    print(f"Saved sample image at epoch {epoch}")
                model.train()


if __name__ == "__main__":  
   train()