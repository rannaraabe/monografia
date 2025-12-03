
import os
import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline

# ======================
# CONFIGURAÇÕES
# ======================
A_PATH = "1961.PNG"
B_PATH = "1962.PNG"
OUTPUT_DIR = "frames_ALTA"
NUM_FRAMES = 20
STRENGTH = 0.45         # 0.15–0.25 recomendado
SEED = 42
WIDTH = 512
HEIGHT = 512

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ======================
# FUNÇÃO PARA CROSSFADER
# ======================
def crossfade(a_img, b_img, alpha):
    arr_a = np.array(a_img).astype(np.float32)
    arr_b = np.array(b_img).astype(np.float32)
    arr = (1 - alpha) * arr_a + alpha * arr_b
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

# ======================
# CARREGA O PIPE img2img
# ======================
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
).to(device)
pipe.safety_checker = None

# ======================
# CARREGA IMAGENS A E B
# ======================
img_a = Image.open(A_PATH).convert("RGB").resize((WIDTH, HEIGHT))
img_b = Image.open(B_PATH).convert("RGB").resize((WIDTH, HEIGHT))

# ======================
# LOOP DE INTERPOLAÇÃO
# ======================
for i, alpha in enumerate(np.linspace(0, 1, NUM_FRAMES)):
    print(f"Gerando frame {i+1}/{NUM_FRAMES} (alpha={alpha:.2f})")

    # 1) Crossfade pixel-a-pixel
    blended = crossfade(img_a, img_b, alpha)

    # 2) Refinar com SD img2img
    result = pipe(
        prompt="a hill covered with Atlantic Forest and a sand dune cutting through it",
        image=blended,
        strength=STRENGTH,
        guidance_scale=0.0,
        num_inference_steps=25,
#        strength = 1.0,
#        guidance_scale = 7,
#        num_inference_steps = 50,
        generator=torch.Generator(device=device).manual_seed(SEED),
    ).images[0]

    result.save(f"{OUTPUT_DIR}/frame_{i:03d}.png")

print("Concluído!")
