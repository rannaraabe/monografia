import os
from pathlib import Path
from tqdm.auto import tqdm

import cv2
import numpy as np
from skimage import exposure
from PIL import Image
import matplotlib.pyplot as plt

# ======================
# CONFIGURAÇÕES
# ======================
INPUT_DIR = Path("/kaggle/input/new-data")
OUTPUT_DIR = Path("/kaggle/working/processed/new-data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# proporção 2:1 (512x256 ou 256x128)
TARGET_SIZE = (512, 256)  # (width, height)

DENOISE_H = 5
CLAHE_CLIP = 2.0 # default
CLAHE_TILE = (8, 8) # default
SHARPEN_KERNEL = np.array([[0,  -0.25, 0], [-0.25, 2, -0.25], [0,  -0.25, 0]])
EXTS = [".jpg", ".jpeg", ".png"]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ======================
# FUNÇÃO AUXILIARES
# ======================
def list_images(folder: Path):
    files = [p for p in folder.rglob("*") if p.suffix.lower() in EXTS]
    return sorted(files)

def read_rgb(path: Path):
    img_bgr = cv2.imread(str(path))
    if img_bgr is None:
        raise RuntimeError(f"Não foi possível ler a imagem: {path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb

def choose_reference_image():
    reference_path = "/kaggle/input/morro-do-careca/sem-data-2.jpg"
    img = cv2.imread(reference_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # garantir RGB
    print("Imagem de referência para histogram matching:", reference_path)
    return img

def upscale(img):
    h, w = img.shape[:2]
    min_size = 512
    if min(h, w) < min_size:
        scale = min_size / min(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    img = cv2.resize(img, (512, 256), interpolation=cv2.INTER_LANCZOS4)
    return img

def plot():
    # n_show = min(6, len(img_paths))
    n_show = len(img_paths)
    fig, axs = plt.subplots(n_show, 2, figsize=(10, 4 * n_show))
    for i in range(n_show):
        orig = read_rgb(img_paths[i])
        proc_path = OUTPUT_DIR / (img_paths[i].stem + "_processed.jpg")
        proc = cv2.imread(str(proc_path), cv2.IMREAD_COLOR)
        axs[i, 0].imshow(orig)
        axs[i, 0].set_title(f"Original: {img_paths[i].name}")
        axs[i, 0].axis("off")
        axs[i, 1].imshow(proc)
        axs[i, 1].set_title("Resultado")
        axs[i, 1].axis("off")
    plt.tight_layout()
    plt.show()
    
    print("Arquivos salvos em:", OUTPUT_DIR)
    
# ======================
# FUNÇAO DE PRE-PROCESSAMENTO
# ======================
def preprocess_v2(img: np.ndarray, target_size=TARGET_SIZE):
    img = cv2.fastNlMeansDenoisingColored(img, None, h=DENOISE_H, templateWindowSize=7, searchWindowSize=21) # +denoise -detalhes da imagem
    img = cv2.GaussianBlur(img, (3,3), 0) # +kernel +blur

    # contrast limited adaptive histogram equalization
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE)
    img = clahe.apply(img)
    
    img = cv2.filter2D(img, -1, SHARPEN_KERNEL)
    
    return img

# ======================
# LOOP DE PROCESSAMENTO
# ======================
img_paths = list_images(INPUT_DIR)
if len(img_paths) == 0:
    raise RuntimeError(f"Nenhuma imagem encontrada em {INPUT_DIR}")

print(f"Base de dados possui {len(img_paths)} imagens")
# ref_img_rgb = choose_reference_image()
# ref_gray = cv2.cvtColor(ref_img_rgb, cv2.COLOR_RGB2GRAY)

print("Processando imagens e salvando em:", OUTPUT_DIR)
for p in tqdm(img_paths):
    try:
        img_rgb = read_rgb(p)
        # processed_gray = preprocess_v1(img_rgb, ref_gray, target_size=TARGET_SIZE)
        processed_gray = preprocess_v2(img_rgb, target_size=TARGET_SIZE)
        out_name = OUTPUT_DIR / (p.stem + "_processed.jpg")
        cv2.imwrite(str(out_name), processed_gray)
    except Exception as e:
        print("Erro processando", p.name, "->", e)

print("Processamento concluído :)")
plot()
