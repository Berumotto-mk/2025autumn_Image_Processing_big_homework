import base64
import io
from pathlib import Path
from typing import Tuple

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms

# -----------------------------
# Model definitions
# -----------------------------


class GANLoss(nn.Module):
    def __init__(self, lsgan: bool = True):
        super().__init__()
        self.register_buffer("real_label", torch.tensor(1.0))
        self.register_buffer("fake_label", torch.tensor(0.0))
        self.loss = nn.MSELoss() if lsgan else nn.BCEWithLogitsLoss()

    def forward(self, prediction, target_is_real):
        target_tensor = self.real_label if target_is_real else self.fake_label
        target_tensor = target_tensor.expand_as(prediction)
        return self.loss(prediction, target_tensor)


class ResnetBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        layers = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0),
            nn.InstanceNorm2d(dim),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0),
            nn.InstanceNorm2d(dim),
        ]
        self.conv_block = nn.Sequential(*layers)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        out = x + self.conv_block(x)
        return self.relu(out)


class ResnetGenerator(nn.Module):
    def __init__(self, ngf=64, n_blocks=6):
        super().__init__()
        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, ngf, kernel_size=7, padding=0),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True),
        ]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            layers += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(ngf * mult * 2),
                nn.ReLU(True),
            ]

        mult = 2 ** n_downsampling
        for _ in range(n_blocks):
            layers += [ResnetBlock(ngf * mult)]

        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            layers += [
                nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(int(ngf * mult / 2)),
                nn.ReLU(True),
            ]

        layers += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, 3, kernel_size=7, padding=0), nn.Tanh()]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class NLayerDiscriminator(nn.Module):
    def __init__(self, ndf=64, n_layers=3):
        super().__init__()
        sequence = [
            nn.Conv2d(3, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
        ]
        nf_mult = 1
        nf_mult_prev = 1

        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=2, padding=1),
                nn.InstanceNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=1, padding=1),
            nn.InstanceNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1)]
        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        return self.model(x)


class CycleGANModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.netG_A = ResnetGenerator(ngf=64, n_blocks=9)
        self.netG_B = ResnetGenerator(ngf=64, n_blocks=9)
        self.netD_A = NLayerDiscriminator(n_layers=3, ndf=64)
        self.netD_B = NLayerDiscriminator(n_layers=3, ndf=64)
        self.criterionGAN = GANLoss()
        self.criterionCycle = nn.L1Loss()
        self.optimizer_G = torch.optim.Adam(
            list(self.netG_A.parameters()) + list(self.netG_B.parameters()), lr=0.0002, betas=(0.5, 0.999)
        )
        self.optimizer_D = torch.optim.Adam(
            list(self.netD_A.parameters()) + list(self.netD_B.parameters()), lr=0.0002, betas=(0.5, 0.999)
        )

    def forward(self, real_A, real_B):
        fake_B = self.netG_A(real_A)
        rec_A = self.netG_B(fake_B)
        fake_A = self.netG_B(real_B)
        rec_B = self.netG_A(fake_A)
        return [fake_B, rec_A, fake_A, rec_B]


# -----------------------------
# Backend utilities
# -----------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 使用相对当前文件目录的权重路径，避免依赖进程工作目录
LANDSCAPE_WEIGHTS = (Path(__file__).parent / "checkpoint" / "cyclegan-landscape.pt")
FIGURE_WEIGHTS = (Path(__file__).parent / "checkpoint" / "cyclegan-figure.pt")
TRANSFORM = transforms.Compose(
    [
        # 统一推理分辨率到 512×512，降低显存占用
        transforms.Resize(512),
        transforms.CenterCrop(512),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    tensor = TRANSFORM(image.convert("RGB")).unsqueeze(0)
    return tensor.to(DEVICE)


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    tensor = tensor.detach().cpu().squeeze(0)
    tensor = torch.clamp(tensor, -1, 1)
    tensor = (tensor + 1) / 2
    array = (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(array)


def pil_to_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def summarize_palette(image: Image.Image) -> Tuple[str, str]:
    small = image.convert("RGB").resize((32, 32))
    data = np.array(small) / 255.0
    brightness = data.mean()
    green_ratio = data[:, :, 1].mean()
    blue_ratio = data[:, :, 2].mean()

    if brightness < 0.35:
        return "幽夜", "夜雨霜风压墨云"
    if green_ratio > blue_ratio:
        return "山林", "松间一径入烟岚"
    if blue_ratio > 0.45:
        return "江水", "孤舟明月过渔湾"
    return "晨曦", "晓风轻抚远天寒"


# -----------------------------
# FastAPI app
# -----------------------------

app = FastAPI(title="CycleGAN Style Service", description="风景水墨 / 工笔人像 / 图文生成" )
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ink_model: CycleGANModel | None = None
gongbi_model: CycleGANModel | None = None


def _load_model_from(weights_path: Path) -> CycleGANModel | None:
    if not weights_path.exists():
        return None
    model = CycleGANModel()
    state = torch.load(weights_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model


def load_models():
    global ink_model, gongbi_model
    ink_model = _load_model_from(LANDSCAPE_WEIGHTS)
    gongbi_model = _load_model_from(FIGURE_WEIGHTS)


@app.on_event("startup")
def _startup():
    load_models()


@app.post("/api/style-transfer/ink")
async def stylize_ink(image: UploadFile = File(...)):
    if ink_model is None:
        raise HTTPException(status_code=503, detail="水墨模型权重未就绪")

    contents = await image.read()
    try:
        pil_image = Image.open(io.BytesIO(contents))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail="无法解析上传的图像") from exc

    tensor = pil_to_tensor(pil_image)
    try:
        with torch.no_grad():
            if DEVICE.type == "cuda":
                # 半精度推理，进一步降低显存占用
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    fake_B = ink_model.netG_A(tensor)
                    rec_A = ink_model.netG_B(fake_B)
            else:
                fake_B = ink_model.netG_A(tensor)
                rec_A = ink_model.netG_B(fake_B)
    except torch.cuda.OutOfMemoryError:
        # GPU 显存不足，自动降级到 CPU 再试一次
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        cpu = torch.device("cpu")
        ink_model.to(cpu)
        tensor = tensor.to(cpu)
        with torch.no_grad():
            fake_B = ink_model.netG_A(tensor)
            rec_A = ink_model.netG_B(fake_B)

    stylized = pil_to_base64(tensor_to_pil(fake_B))
    cycle = pil_to_base64(tensor_to_pil(rec_A))
    return {"stylized": stylized, "cycle": cycle}


@app.post("/api/style-transfer/gongbi")
async def stylize_gongbi(image: UploadFile = File(...)):
    if gongbi_model is None:
        raise HTTPException(status_code=503, detail="工笔模型权重未就绪")

    contents = await image.read()
    try:
        pil_image = Image.open(io.BytesIO(contents))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail="无法解析上传的图像") from exc

    tensor = pil_to_tensor(pil_image)
    try:
        with torch.no_grad():
            if DEVICE.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    fake_B = gongbi_model.netG_A(tensor)
                    rec_A = gongbi_model.netG_B(fake_B)
            else:
                fake_B = gongbi_model.netG_A(tensor)
                rec_A = gongbi_model.netG_B(fake_B)
    except torch.cuda.OutOfMemoryError:
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        cpu = torch.device("cpu")
        gongbi_model.to(cpu)
        tensor = tensor.to(cpu)
        with torch.no_grad():
            fake_B = gongbi_model.netG_A(tensor)
            rec_A = gongbi_model.netG_B(fake_B)

    stylized = pil_to_base64(tensor_to_pil(fake_B))
    cycle = pil_to_base64(tensor_to_pil(rec_A))
    return {"stylized": stylized, "cycle": cycle}


@app.post("/api/poem")
async def generate_poem(image: UploadFile = File(...)):
    contents = await image.read()
    try:
        pil_image = Image.open(io.BytesIO(contents))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail="无法解析上传的图像") from exc

    theme, line = summarize_palette(pil_image)
    poem = f"{theme}含章影，{line}"
    return {"poem": poem}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="0.0.0.0", port=8001, reload=False)
