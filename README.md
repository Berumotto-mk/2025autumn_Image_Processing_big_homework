

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
