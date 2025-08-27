import os
from pathlib import Path
import time
from io import BytesIO
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from PIL import Image
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def load_image(path, size):
    img = Image.open(path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor()
    ])
    return transform(img).unsqueeze(0).to(device)
def save_image(tensor, out_path):
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    img = tensor.detach().cpu().squeeze(0).clamp(0,1)
    arr = (img.mul(255).permute(1,2,0).byte().numpy())
    pil = Image.fromarray(arr)
    buf = BytesIO()
    pil.save(buf, format="JPEG")
    data = buf.getvalue()
    tmp = None
    for attempt in range(6):
        try:
            import tempfile
            with tempfile.NamedTemporaryFile(dir=str(p.parent), delete=False, suffix=".tmp.jpg") as f:
                tmp = f.name
                f.write(data)
                f.flush()
            os.replace(tmp, str(p))
            print(f"Saved: {p} (attempt {attempt+1})")
            return
        except Exception as e:
            print("save attempt failed:", e)
            time.sleep(0.1 * (attempt+1))
            try:
                if tmp and Path(tmp).exists():
                    Path(tmp).unlink()
            except Exception:
                pass
    raise OSError(f"Failed to save {p}")
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = torch.tensor(mean).view(-1,1,1).to(device)
        self.std  = torch.tensor(std).view(-1,1,1).to(device)
    def forward(self, img):
        return (img - self.mean) / self.std
class GramMatrix(nn.Module):
    def forward(self, x):
        b,c,h,w = x.size()
        feats = x.view(b,c,h*w)
        G = torch.bmm(feats, feats.transpose(1,2))
        return G / (c*h*w)
class ContentLoss(nn.Module):
    def __init__(self, target):
        super().__init__()
        self.target = target.detach()
        self.loss = 0.0
    def forward(self, x):
        self.loss = nn.functional.mse_loss(x, self.target)
        return x
class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super().__init__()
        self.target = GramMatrix()(target_feature).detach()
        self.loss = 0.0
    def forward(self, x):
        G = GramMatrix()(x)
        self.loss = nn.functional.mse_loss(G, self.target)
        return x

def get_style_model_and_losses(cnn, mean, std, style_img, content_img,
                               content_layer='conv_4',
                               style_layers=('conv_1','conv_2','conv_3','conv_4','conv_5')):
    cnn = cnn.features.eval().to(device)
    normalization = Normalization(mean, std).to(device)
    model = nn.Sequential(normalization)
    style_losses = []
    content_losses = []
    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f"conv_{i}"
        elif isinstance(layer, nn.ReLU):
            name = f"relu_{i}"
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f"pool_{i}"
        elif isinstance(layer, nn.BatchNorm2d):
            name = f"bn_{i}"
        else:
            name = f"layer_{i}"
        model.add_module(name, layer)
        if name == content_layer:
            target = model(content_img).detach()
            cl = ContentLoss(target)
            model.add_module(f"content_loss_{i}", cl)
            content_losses.append(cl)
        if name in style_layers:
            target_feature = model(style_img).detach()
            sl = StyleLoss(target_feature)
            model.add_module(f"style_loss_{i}", sl)
            style_losses.append(sl)
    # trim after last loss layer
    for j in range(len(model)-1, -1, -1):
        if isinstance(model[j], (ContentLoss, StyleLoss)):
            break
    model = model[:j+1]
    return model, style_losses, content_losses

def run_style_transfer(cnn, content_img, style_img, input_img,
                       num_steps=50, style_weight=1e6, content_weight=1):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    model, style_losses, content_losses = get_style_model_and_losses(
        cnn, mean, std, style_img, content_img)
    optimizer = optim.Adam([input_img.requires_grad_()], lr=0.02)
    for step in range(num_steps):
        optimizer.zero_grad()
        model(input_img)
        style_score = sum(sl.loss for sl in style_losses)
        content_score = sum(cl.loss for cl in content_losses)
        loss = style_score * style_weight + content_score * content_weight
        loss.backward()
        optimizer.step()
        if step % 10 == 0 or step == num_steps-1:
            print(f"Step {step}: style {style_score.item():.2f}, content {content_score.item():.2f}")
    return input_img.detach()

if __name__ == "__main__":
    torch.set_num_threads(os.cpu_count() or 4)
    content_path = r"C:\Users\somai\OneDrive\Documents\Process1\content.jpg"
    style_path   = r"C:\Users\somai\OneDrive\Documents\Process1\style.jpg"
    out_path     = r"C:\Users\somai\OneDrive\Documents\Process1\stylized_result.jpg"
    p_content = Path(content_path)
    p_style   = Path(style_path)
    if not p_content.exists() or not p_style.exists():
        raise FileNotFoundError("Place content.jpg and style.jpg in the Process1 folder.")
    imsize = 128
    steps = 50
    content = load_image(content_path, imsize)
    style = load_image(style_path, imsize)
    input_img = content.clone().to(device)
    cnn = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
    output = run_style_transfer(cnn, content, style, input_img, num_steps=steps)
    save_image(output, out_path)
    print("Done. Saved", out_path)