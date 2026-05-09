"""
GUI for sketch-to-photo diffusion model.
Run: python gui.py
Open: http://localhost:7860 in your browser
"""

import torch
import math
import os
import glob
from pathlib import Path
import gradio as gr
from PIL import Image
import torchvision.transforms as T

from u_net import ConditionalUNet, offset_cosine_diffusion_schedule


# ---------- setup ----------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

CHECKPOINT_DIR = "checkpoints"
IMAGE_SIZE = 256

# transforms
to_tensor = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize([0.5]*3, [0.5]*3),
])

# cache for loaded models
model_cache = {}


def get_available_epochs():
    """List all checkpoint epochs available."""
    files = sorted(glob.glob(f"{CHECKPOINT_DIR}/ckpt_epoch*.pt"))
    epochs = []
    for f in files:
        name = os.path.basename(f)
        epoch_num = int(name.replace("ckpt_epoch", "").replace(".pt", ""))
        epochs.append(epoch_num)
    return epochs


def load_model_for_epoch(epoch, use_ema=False):
    """Load (or fetch from cache) the model for a given epoch."""
    cache_key = (epoch, use_ema)
    if cache_key in model_cache:
        return model_cache[cache_key]
    
    ckpt_path = f"{CHECKPOINT_DIR}/ckpt_epoch{epoch:04d}.pt"
    print(f"Loading {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    
    model = ConditionalUNet(image_size=IMAGE_SIZE, noise_embedding_size=64).to(device)
    weights_key = "ema_model" if use_ema else "model"
    model.load_state_dict(ckpt[weights_key])
    model.eval()
    
    model_cache[cache_key] = model
    return model


@torch.no_grad()
def generate(sketch_image, epoch, num_steps, use_ema, seed):
    """Generate a photo from a sketch."""
    if sketch_image is None:
        return None, "Please upload a sketch first."
    
    try:
        epoch = int(epoch)
        num_steps = int(num_steps)
        seed = int(seed) if seed else None
    except (ValueError, TypeError):
        return None, "Invalid epoch or num_steps value."
    
    # set seed for reproducibility
    if seed is not None:
        torch.manual_seed(seed)
    
    # load model
    try:
        model = load_model_for_epoch(epoch, use_ema)
    except FileNotFoundError:
        return None, f"Checkpoint for epoch {epoch} not found."
    
    # preprocess sketch
    if isinstance(sketch_image, dict):
        sketch_image = sketch_image.get("composite", sketch_image)
    sketch_pil = sketch_image if isinstance(sketch_image, Image.Image) else Image.fromarray(sketch_image)
    sketch_pil = sketch_pil.convert("RGB")
    sketch = to_tensor(sketch_pil).unsqueeze(0).to(device)
    
    # diffusion sampling
    x = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)
    times = torch.linspace(1.0 - 1e-3, 1e-3, num_steps + 1, device=device)
    
    for i in range(num_steps):
        t      = times[i].view(1, 1, 1, 1)
        t_next = times[i + 1].view(1, 1, 1, 1)
        
        nr,  sr  = offset_cosine_diffusion_schedule(t)
        nrn, srn = offset_cosine_diffusion_schedule(t_next)
        
        pn = model(x, sketch, t)
        pi = (x - nr * pn) / sr
        x = srn * pi + nrn * pn
    
    # convert from [-1, 1] to [0, 1] for display
    output = (x.clamp(-1, 1) + 1) / 2
    output_np = output.squeeze().permute(1, 2, 0).cpu().numpy()
    output_np = (output_np * 255).clip(0, 255).astype("uint8")
    
    return output_np, f"Generated using epoch {epoch} {'(EMA)' if use_ema else '(model)'}, {num_steps} steps."


# ---------- build the UI ----------

available_epochs = get_available_epochs()
default_epoch = available_epochs[-1] if available_epochs else 50

with gr.Blocks(title="Sketch 2 Real Diffusion") as demo:
    gr.Markdown("# Sketch 2 Real")
    gr.Markdown("Upload a sketch and the model generates a photo from it.")
    
    with gr.Row():
        with gr.Column():
            sketch_input = gr.Image(
                label="Upload sketch",
                type="pil",
                height=300,
            )
            
            epoch_input = gr.Dropdown(
                choices=available_epochs,
                value=default_epoch,
                label="Checkpoint epoch",
                info=f"Available: {available_epochs[0]} to {available_epochs[-1]}" if available_epochs else "No checkpoints found",
            )
            
            steps_slider = gr.Slider(
                minimum=10,
                maximum=500,
                value=200,
                step=10,
                label="Number of sampling steps",
                info="More steps = better quality but slower",
            )
            
            ema_checkbox = gr.Checkbox(
                value=False,
                label="Use EMA weights",
                info="EMA usually only useful for late-training checkpoints",
            )
            
            seed_input = gr.Number(
                value=42,
                label="Random seed (for reproducibility)",
                precision=0,
            )
            
            generate_btn = gr.Button("Generate", variant="primary")
        
        with gr.Column():
            output_image = gr.Image(label="Generated photo", height=400)
            output_text = gr.Textbox(label="Status", interactive=False)
    
    generate_btn.click(
        fn=generate,
        inputs=[sketch_input, epoch_input, steps_slider, ema_checkbox, seed_input],
        outputs=[output_image, output_text],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)