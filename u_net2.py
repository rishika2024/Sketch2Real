import torch
from PIL import Image
import torchvision.transforms as T
import math
import random
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
import copy
from torchvision.utils import save_image
from tqdm import tqdm

def cosine_diffusion_schedule(t):
    # signal_rate = cos(t * pi/2), noise_rate = sin(t * pi/2)
    # t = normalized diffusion timestep [0, 1]
    # torch.cos and torch.sin work for tensor inputs
    signal_rates = torch.cos(t * math.pi / 2) 
    noise_rates  = torch.sin(t * math.pi / 2)
    return noise_rates, signal_rates


def add_noise(colored_image, noise_rates, signal_rates):
    # forward process: x_t = signal_rate * x_0 + noise_rate * noise(epsilon)
    noise = torch.randn_like(colored_image) # create noise with same shape as colored_image
    noisy_image = signal_rates * colored_image + noise_rates * noise
    return noisy_image, noise

def sinusoidal_embedding(t, embedding_dim=32):
    # build a vector of frequencies that are log-uniformly spaced between 1 and 1000
    # angular speed = 2 * pi * frequency, so that we get full cycles at different rates
    # and then concatenate sin and cos of these angular speeds times t to get the embedding
    frequencies = torch.exp(torch.linspace(math.log(1.0), math.log(1000.0), embedding_dim // 2, device=t.device))
    angular_speeds = 2.0 * math.pi * frequencies
    embeddings = torch.cat([torch.sin(angular_speeds * t), torch.cos(angular_speeds * t)], dim=1)
    return embeddings


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # since we add the input to the output, they must have the same number of channels
        # output = conv2(SILU(conv1(batch_norm(input)))) + residual(input)
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
        x = F.silu(self.conv1(x))
        x = self.conv2(x)
        return x + residual


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, block_depth):
        super().__init__()
        self.block_depth = block_depth
        # list of residual blocks
        # the first block goes from in_channels to out_channels
        # and the rest go from out_channels to out_channels
        self.res_blocks = nn.ModuleList() 
        for i in range(block_depth):
            if i == 0:
                self.res_blocks.append(ResidualBlock(in_channels, out_channels))
            else:
                self.res_blocks.append(ResidualBlock(out_channels, out_channels))

    def forward(self, x, skips):
        # for each residual block
        # input_i+1 = residualblock(input_i)
        # output = avg_pool2d(final_residual_output)
        # skips = list of outpiuts of each residual block
        for block in self.res_blocks:
            x = block(x)
            skips.append(x)
        x = F.avg_pool2d(x, kernel_size=2) # downsample by 2x
        return x


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, block_depth, skip_channels):
        super().__init__()
        self.block_depth = block_depth
        # list of residual blocks
        # the first block goes from in_channels to out_channels
        # and the rest go from out_channels to out_channels
        self.res_blocks = nn.ModuleList()        
        for i in range(block_depth):
            if i == 0:
                in_ch = in_channels + skip_channels
            else:
                in_ch = out_channels + skip_channels
            self.res_blocks.append(ResidualBlock(in_ch, out_channels))

    def forward(self, x, skips):
        x = F.interpolate(x, scale_factor=2, mode='nearest') # upsample by 2x
        for block in self.res_blocks:
            # for each residual block
            # input_i+1 = residualblock(concat(input_i, corresponding_skip))
            skip = skips.pop() # get the corresponding skip connection from the downsampling path
            x = torch.cat([x, skip], dim=1) # concatenate along channel dimension
            x = block(x)  
        return x


class ConditionalUNet(nn.Module):
    def __init__(self, image_size=128, noise_embedding_size=64):
        super().__init__()
        self.image_size = image_size
        self.noise_embedding_size = noise_embedding_size 
        
        # since noisy image and sketch are concatenated, the input channels = 3 (image) + 3 (sketch) = 6
        self.initial_conv = nn.Conv2d(6, 64, kernel_size=1) 
        # no. of channels = 64(initial conv) + noise_embedding_size (after concatenating noise embedding) = 128
        in_after_concat = 64 + noise_embedding_size

        # downsampling path feature channels: 128 -> 32 -> 64 -> 128
        self.down1 = DownBlock(in_after_concat, 32,  block_depth=2)
        self.down2 = DownBlock(32,  64, block_depth=2)
        self.down3 = DownBlock(64, 128, block_depth=2)
        
        # bottleneck path feature channels: 128 -> 256 -> 256 -> 128
        self.bottleneck1 = ResidualBlock(128, 256)
        self.bottleneck2 = ResidualBlock(256, 256)
        self.bottleneck3 = ResidualBlock(256, 128)

        # upsampling path feature channels: 128 -> 64 -> 32 -> 16
        self.up1 = UpBlock(128, 64, block_depth=2, skip_channels=128)
        self.up2 = UpBlock(64, 32,  block_depth=2, skip_channels=64)
        self.up3 = UpBlock(32,  16,  block_depth=2, skip_channels=32)
        
        # final conv to get back to 3 channels (RGB)
        self.final_conv = nn.Conv2d(16, 3, kernel_size=1)

        # initialize final conv's weight and bias to zero
        # so that at the start of training, the model just predicts noise = 0 and x_t = x_0
        nn.init.zeros_(self.final_conv.weight)
        nn.init.zeros_(self.final_conv.bias)

    def forward(self, noisy_images, sketches, noise_variances):
        # concatenate noisy_images and sketches along the channel dimension
        # this is so that the model can condition on the sketch when predicting the noise
        x = torch.cat([noisy_images, sketches], dim=1)
        # initial conv to get to 64 channels
        x = self.initial_conv(x)
        
        # create noise embedding
        noise_emb = sinusoidal_embedding(noise_variances, self.noise_embedding_size)
        # reshape tensor
        # noise_emb shape: (B, noise_embedding_size) -> (B, noise_embedding_size, 1, 1) -> (B, noise_embedding_size, image_size, image_size)
        noise_emb = noise_emb.view(-1, self.noise_embedding_size, 1, 1)      
        noise_emb = F.interpolate(noise_emb, size=(self.image_size, self.image_size), mode="nearest")
        
        # concatenate noise embedding to the feature maps after the initial conv
        x = torch.cat([x, noise_emb], dim=1)

        skips = [] # list to store skip connections for the upsampling path

        # downsampling path with skip connections
        x = self.down1(x, skips) 
        x = self.down2(x, skips)
        x = self.down3(x, skips)
        
        # bottleneck
        x = self.bottleneck1(x)
        x = self.bottleneck2(x)
        x = self.bottleneck3(x)

        # upsampling path with skip connections
        x = self.up1(x, skips)
        x = self.up2(x, skips)
        x = self.up3(x, skips)
        
        # return final conv to get back to 3 channels (RGB)
        return self.final_conv(x)


def update_ema(ema_model, model, decay=0.999):    
    with torch.no_grad(): # no gradients descent for EMA model
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)


class SketchPhotoDataset(Dataset):
    """Loads (sketch, photo) pairs. Filenames match across both folders."""
    def __init__(self, image_dir, sketch_dir, split='train', val_fraction=0.1, seed=42):
        # after shuffle, val = 10% and train = 90% of the data
        self.image_dir  = Path(image_dir)
        self.sketch_dir = Path(sketch_dir)  
        
        # get all filenames and sort so that we have matching pairs
        all_files = sorted([f.name for f in self.image_dir.iterdir()])            
        
       # shuffle and split for test and val
        rng = random.Random(seed)
        rng.shuffle(all_files)
        n_val = int(len(all_files) * val_fraction)
        if split == 'train':
            self.filenames = all_files[n_val:]
        else:  # val
            self.filenames = all_files[:n_val]

        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        name   = self.filenames[idx]
        photo  = Image.open(self.image_dir  / name).convert("RGB")
        sketch = Image.open(self.sketch_dir / name).convert("RGB")
        return self.transform(sketch), self.transform(photo)


def train(
    image_dir       = "dataset_small",
    sketch_dir      = "sketch_small",
    output_dir      = "checkpoints_small",
    image_size      = 128,
    batch_size      = 64,
    epochs          = 500,
    lr              = 1e-3,
    weight_decay    = 1e-4,
    ema_decay       = 0.999,
    noise_embedding_size = 64,
    save_every      = 10,
    val_fraction    = 0.1,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    """SPLIT TEST AND VAL DATASET"""
    train_dataset = SketchPhotoDataset(image_dir, sketch_dir, split='train', val_fraction=val_fraction)
    val_dataset   = SketchPhotoDataset(image_dir, sketch_dir, split='val',   val_fraction=val_fraction)
    
    """LOAD DATA"""
    # num_workers = number of cpu threads for loading data.
    # pin_memory =  if True, the data loader will copy Tensors into CUDA pinned memory before returning them. (for external gpu)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True, drop_last=True)
    
    print(f"Train: {len(train_dataset)} pairs, Val: {len(val_dataset)} pairs")

    model = ConditionalUNet(image_size=image_size, noise_embedding_size=noise_embedding_size).to(device)
    # p.numel() gives the number of parameters in the model
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # create EMA model as a copy of the original model
    #  Not updating the weights with gradients
    ema_model = copy.deepcopy(model)
    for p in ema_model.parameters():
        p.requires_grad_(False)
    
    # using AdamW optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    """TRAINING LOOP"""

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for sketches, photos in pbar:
            sketches = sketches.to(device)
            photos = photos.to(device)
            B  = photos.shape[0] # batch size 
            
            # diffusion step t is randomly sampled for each image in the batch
            #  compute the corresponding noise and signal rates using the cosine schedule
            t = torch.rand(B, 1, 1, 1, device=device) 
            noise_rates, signal_rates = cosine_diffusion_schedule(t)
            noisy_photos, noise = add_noise(photos, noise_rates, signal_rates)
                        
            noise_variances = noise_rates ** 2
            predicted_noise = model(noisy_photos, sketches, noise_variances)
            
            # using L1 loss
            loss = F.l1_loss(predicted_noise, noise)

            optimizer.zero_grad() # reset gradients to zero before backpropagation
            loss.backward() # compute gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # gradient clipping to prevent exploding gradients
            optimizer.step() # update model weights based on computed gradients

            update_ema(ema_model, model, decay=ema_decay) # update EMA model weights

            epoch_loss += loss.item() 
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = epoch_loss / len(train_loader)

        """VALIDATION LOOP"""
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for sketches, photos in val_loader:
                sketches, photos = sketches.to(device), photos.to(device)
                B = photos.shape[0]
                t = torch.rand(B, 1, 1, 1, device=device)
                noise_rates, signal_rates = cosine_diffusion_schedule(t)
                noisy_photos, noise = add_noise(photos, noise_rates, signal_rates)
                predicted_noise = model(noisy_photos, sketches, noise_rates ** 2)
                val_loss += F.l1_loss(predicted_noise, noise).item()
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch} | train loss: {avg_loss:.4f} | val loss: {val_loss:.4f}")

        # save checkpoint
        if epoch % save_every == 0 or epoch == epochs:
            ckpt = {
                "epoch":     epoch,
                "model":     model.state_dict(),
                "ema_model": ema_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "avg_loss":  avg_loss,
                "val_loss":  val_loss,
            }
            torch.save(ckpt, output_dir / f"ckpt_epoch{epoch:04d}.pt")
            print(f"Saved checkpoint at epoch {epoch}")

            # generate a sample using a VAL sketch
            with torch.no_grad():
                sample_sketch, _ = val_dataset[0]
                sample_sketch = sample_sketch.unsqueeze(0).to(device)
                
                # pure noise image to start the reverse diffusion process
                x = torch.randn(1, 3, image_size, image_size, device=device)
                num_steps = 200

                # diffusion timesteps from 1 to 0
                times = torch.linspace(1.0 - 1e-3, 1e-3, num_steps + 1, device=device)
                
                for i in range(num_steps):
                    t = times[i].view(1, 1, 1, 1)
                    t_next = times[i + 1].view(1, 1, 1, 1)
                    
                    noise_rate, signal_rate = cosine_diffusion_schedule(t)
                    noise_rate_next, signal_rate_next = cosine_diffusion_schedule(t_next)
                    noise_variances = noise_rate ** 2
                    
                    # predict the noise using the model
                    # calculate the predicted image at the current timestep using the predicted noise and the noisy image
                    predicted_noise = model(x, sample_sketch, noise_variances)
                    predicted_image = (x - noise_rate * predicted_noise) / signal_rate
                    x = signal_rate_next * predicted_image + noise_rate_next * predicted_noise
                
                # concatenate the input sketch and the generated image for visualization
                combined = torch.cat([sample_sketch, x.clamp(0, 1)], dim=0)
                save_image(combined, output_dir / f"sample_epoch{epoch:04d}.png", nrow=2)
                print(f"Saved sample image at epoch {epoch}")


if __name__ == "__main__":  
   train()