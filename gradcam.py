"""
Grad-CAM 终版 v4 - 完美适配 RETFound + LoRA
"""
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import os, glob, random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pytorch_grad_cam.utils.image import show_cam_on_image
from models.retfound_classifier import RETFoundClassifier

# ==================== 配置 ====================
CKPT_PATH    = "runs/20261009_120703/best.pt"
OUTPUT_DIR   = "outputs/gradcam"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
CLASS_NAMES  = ["0_No_DR", "1_Mild_NPDR", "2_Moderate_NPDR", "3_Severe_NPDR", "4_Proliferative_DR"]
DATA_ROOT    = "data/DDR_real"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================== 构建模型 ====================
print("Building model...")
model = RETFoundClassifier(
    num_classes=5,
    weights_path=None,  # 不重新加载 RETFound，下一步用 checkpoint 覆盖
    img_size=224,
    dropout=0.1,
    lora_r=8,
    lora_alpha=16,
    lora_dropout=0.05,
)

# 加载完整 checkpoint
print("Loading checkpoint...")
ckpt = torch.load(CKPT_PATH, map_location="cpu")
state = ckpt["model_state"]
print(f"  QWK={ckpt['metrics'].get('qwk', 0):.4f}, Acc={ckpt['metrics'].get('accuracy', 0):.4f}")

# 兼容前缀（checkpoint 已经是 "backbone.xxx" 格式，但 RETFound 内部也是 backbone）
missing, unexpected = model.load_state_dict(state, strict=False)
print(f"  missing={len(missing)}, unexpected={len(unexpected)}")
if unexpected:
    print(f"  unexpected 示例: {unexpected[:5]}")

model = model.to(DEVICE)
model.eval()

# ==================== 找 5 类图片 ====================
print("\nFinding 5 class images...")
samples = {}
for cls in range(5):
    cls_dir = os.path.join(DATA_ROOT, CLASS_NAMES[cls])
    imgs = sorted(glob.glob(os.path.join(cls_dir, "*.jpg")))
    if imgs:
        samples[cls] = random.choice(imgs)
        print(f"  Class {cls} ({CLASS_NAMES[cls]}): {len(imgs)} images, picked {os.path.basename(samples[cls])}")

# ==================== 预处理 ====================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ==================== 测试推理 ====================
print("\nTesting inference...")
test_path = list(samples.values())[0]
test_img = Image.open(test_path).convert("RGB")
test_tensor = transform(test_img).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    test_logits = model(test_tensor)
    test_probs = torch.softmax(test_logits, dim=1)[0]
    print(f"  Logits: {test_logits[0].cpu().numpy().round(3)}")
    print(f"  Probs:  {test_probs.cpu().numpy().round(3)}")
    print(f"  Pred: Class {torch.argmax(test_probs).item()}, Conf: {test_probs.max().item():.2%}")

# ==================== Grad-CAM ====================
print("\nGenerating heatmaps...")

def get_vit_cam(model, input_tensor, target_class):
    """对 ViT backbone 最后一个 block 的 norm2 做 Grad-CAM"""
    activations = {}
    gradients = {}
    
    def f_hook(m, i, o): activations['v'] = o.detach()
    def b_hook(m, gi, go): gradients['v'] = go[0].detach()
    
    target = model.backbone.blocks[-1].norm2
    h1 = target.register_forward_hook(f_hook)
    h2 = target.register_full_backward_hook(b_hook)
    
    try:
        model.zero_grad()
        logits = model(input_tensor)
        logits[0, target_class].backward(retain_graph=True)
        
        act = activations['v']   # [1, 197, 1024]
        grad = gradients['v']    # [1, 197, 1024]
        w = grad.mean(dim=1, keepdim=True)  # [1, 1, 1024]
        cam = (act * w).sum(-1)[0, 1:].cpu().numpy()  # 去掉 cls token
        cam = np.maximum(cam, 0)
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cam = cam.reshape(14, 14)
        cam = cv2.resize(cam, (224, 224))
        return cam
    finally:
        h1.remove()
        h2.remove()

# ==================== 画图 ====================
fig, axes = plt.subplots(5, 3, figsize=(15, 25))

for cls, img_path in sorted(samples.items()):
    print(f"  Class {cls}: {os.path.basename(img_path)}")
    
    img = Image.open(img_path).convert("RGB")
    img_resized = img.resize((224, 224))
    rgb_img = np.array(img_resized) / 255.0
    input_tensor = transform(img_resized).unsqueeze(0).to(DEVICE)
    
    cam = get_vit_cam(model, input_tensor, cls)
    vis = show_cam_on_image(rgb_img, cam, use_rgb=True)
    
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)[0]
        pred_cls = torch.argmax(probs).item()
        pred_conf = probs[pred_cls].item()
    
    axes[cls, 0].imshow(img_resized)
    axes[cls, 0].set_title(f"Class {cls}: {CLASS_NAMES[cls]}", fontsize=11)
    axes[cls, 0].axis("off")
    
    axes[cls, 1].imshow(cam, cmap="jet", vmin=0, vmax=1)
    axes[cls, 1].set_title(f"Heatmap (max={cam.max():.3f}, mean={cam.mean():.3f})", fontsize=11)
    axes[cls, 1].axis("off")
    
    axes[cls, 2].imshow(vis)
    axes[cls, 2].set_title(f"Overlay (Pred:{pred_cls}, Conf:{pred_conf:.1%})", fontsize=11)
    axes[cls, 2].axis("off")

plt.suptitle("DR 5-Class Grad-CAM (RETFound-MAE + LoRA)", fontsize=16)
plt.tight_layout()
save_path = os.path.join(OUTPUT_DIR, "gradcam_5classes.png")
plt.savefig(save_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved: {save_path}")

# 单图保存
print("\nSaving individual heatmaps...")
for cls, img_path in sorted(samples.items()):
    img = Image.open(img_path).convert("RGB")
    img_resized = img.resize((224, 224))
    rgb_img = np.array(img_resized) / 255.0
    input_tensor = transform(img_resized).unsqueeze(0).to(DEVICE)
    
    cam = get_vit_cam(model, input_tensor, cls)
    vis = show_cam_on_image(rgb_img, cam, use_rgb=True)
    
    single_path = os.path.join(OUTPUT_DIR, f"gradcam_class{cls}.jpg")
    cv2.imwrite(single_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    print(f"  {single_path}")

print("\nAll done!")
