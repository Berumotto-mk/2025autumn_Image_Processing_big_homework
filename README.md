

文件结构  
.
├── cyclegan_figure  
│   ├── checkpoint  
│   └── data  
│       ├── dataA  
│       │   └── train  
│       └── dataB  
│           └── train  
├── cyclegan_landscape  
│   ├── checkpoint  
│   ├── data  
│   │   ├── dataA  
│   │   │   └── trainA  
│   │   └── dataB  
│   │       └── trainB  
│   └── __pycache__  
└── Front_Back  
    ├── checkpoint  
    ├── __pycache__  
    └── web  

数据集合下载链接

环境配置

- 系统：Linux（已在 Ubuntu 上使用）
- Python：建议 3.12（与 PyTorch、FastAPI 兼容性较好）
- 可选 GPU：NVIDIA 驱动 + CUDA（使用 Conda 的 `pytorch-cuda` 元包更省心）

必需依赖
- 后端与训练/推理：fastapi、uvicorn、pillow、numpy、torch、torchvision、tqdm、matplotlib
- 可选（公网访问）：cloudflared（用于 Cloudflare Tunnel）

一键创建 Conda 环境（推荐）
```bash
# 1) 创建并激活环境
conda create -n transfer python=3.12 -y
conda activate transfer

# 2) 安装 PyTorch（GPU，CUDA 12.8 示例）

# 若仅 CPU：
# conda install -y pytorch torchvision cpuonly -c pytorch

# 3) 安装其余依赖
pip install fastapi "uvicorn[standard]" pillow numpy tqdm matplotlib

# 4) 可选：cloudflared（用于公网隧道）
# Ubuntu/Debian 可直接下载二进制放入 /usr/local/bin，或参考官方文档：
# https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/install-and-setup/installation
```

使用 pip（可选方案）
```bash
python -m venv .venv && source .venv/bin/activate

# PyTorch（根据你的 GPU/CPU 选择对应指令）
# CPU 版本：
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
# CUDA 12.8 例子：
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 其余依赖
pip install fastapi "uvicorn[standard]" pillow numpy tqdm matplotlib
```

模型权重准备
- 风景水墨风格权重：`Front_Back/checkpoint/cyclegan-landscape.pt`
- 工笔人物风格权重：`Front_Back/checkpoint/cyclegan-figure.pt`
（仓库已包含示例权重文件名，如缺失请按上述路径放置）

运行

```bash
# 进入后端目录并启动 FastAPI 服务
cd Front_Back
python server.py  # 服务默认监听 http://localhost:8001
```

前端使用

- 打开 Front_Back/web/index.html（浏览器直接打开或通过本地静态服务器）。
- 模块「A + C · 水墨明信片」支持：
    - 上传风景图 → 自动生成水墨图（A）+ 诗句（C）。
    - 页面内预览明信片排版，可一键下载 PNG。

说明


- 水墨风格使用 Front_Back/checkpoint/cyclegan-landscape.pt。
- 工笔画人物风格使用 Front_Back/checkpoint/cyclegan-figure.pt
- 使用cyclegan_figure下的test.py要传入cyclegan_figure/checkpoint/cyclegan-figure.pt
- 使用cyclegan_landscape下的test.py cyclegan_landscape/checkpoint/cyclegan-landscape.pt
- 诗句生成当前基于图像色调的简易规则，可在后端 /api/poem 替换为更强的图文模型

可选：公网访问（Cloudflare Tunnel）
- 临时链接（Quick Tunnel）：
```bash
cloudflared tunnel --url http://localhost:8000 --no-autoupdate
```
- 稳定域名（命名 Tunnel，需有托管到 Cloudflare 的自有域名）：
    1) `cloudflared login` 完成授权；
    2) `cloudflared tunnel create transfer-web` 并记录 UUID；
    3) 绑定子域名：`cloudflared tunnel route dns transfer-web web.<你的域名>`；
    4) 在 `~/.cloudflared/config.yml` 指定 `tunnel`、`credentials-file` 与 `ingress` 映射到 `http://localhost:8000`；
    5) 启动：`cloudflared tunnel run transfer-web`。
