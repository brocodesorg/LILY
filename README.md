# 🌸 LILY — 3D Procedural Botanical Animation in Python

> **A cinematic 3D procedural animation of a bouquet of Oriental Lilies in a cut-crystal goblet.**  
> Built from scratch with pure mathematical geometry in Python, rendered in vertical **9:16 (1080 × 1920)** for Instagram Reels, YouTube Shorts, and TikTok.

---

## ✨ Features & Visual Highlights

- **🌸 Procedural Botanical Geometry:** Parametric Bezier & spline curves generating petals, stamens, pistils, pollen anthers, and curling veins.
- **💎 Cut-Crystal Goblet & Gold Band:** Ray-traced refraction highlights, caustic glints, and an ornate embossed gold collar.
- **✨ Subsurface Scattering & Studio Lighting:** Soft organic light penetration through translucent petals with velvety shading.
- **🌿 Micro-Atmospheric Details:** Dewdrops with internal fresnel reflections, fallen satin folds, scattered petals, and dreamy background bokeh fairy lights.
- **📱 9:16 Vertical Reel Format:** Native 1080 × 1920 at 30 FPS, optimized for high-retention short-form video algorithms.
- **⚡ Multiprocess Rendering Engine:** Vectorized NumPy operations parallelized across CPU cores with high-bitrate FFmpeg encoding.

---

## 🎬 Story & Animation Progression

```
 0.0s –  3.5s   Goblet & stems materialize; delicate vine curves trace into 3D space.
 3.5s –  8.0s   Petal meshes unfurl progressively with organic blooming curves.
 8.0s – 11.5s   Stamens and pistils form; dew crystals and fairy bokeh lights illuminate.
11.5s – 13.0s   Cinematic camera orbit around the fully bloomed bouquet in crystal glory.
```

---

## 📁 Repository Structure

```text
LILY/
├── generate_lily_animation.py      # Core procedural 3D math & render engine
├── lily_3d_botanical_animation.mp4 # Full 1080x1920 rendered animation output
├── requirements.txt                # Python dependencies (numpy, opencv-python)
├── .gitignore                      # Clean repository ignore rules
└── README.md                       # Documentation & showcase
```

---

## 🚀 Quick Start

### 1. Requirements

Ensure you have **Python 3.9+** and **FFmpeg** installed on your system.

```bash
# Clone the repository
git clone https://github.com/brocodesorg/LILY.git
cd LILY

# Install Python dependencies
pip install -r requirements.txt
```

> **Note on FFmpeg:** FFmpeg is used to assemble frames into the high-bitrate `.mp4`. On Windows, install via `winget install Gyan.FFmpeg` or download from [ffmpeg.org](https://ffmpeg.org).

### 2. Run the Generator

To generate the complete 13-second 1080 × 1920 animation:

```bash
python generate_lily_animation.py
```

*Or open the folder in VS Code and click the **▶ Run** button.*

---

## ⚙️ Technical Specifications

| Parameter | Value |
|---|---|
| **Resolution** | 1080 × 1920 (9:16 Vertical) |
| **Frame Rate** | 30 FPS |
| **Duration** | 13.0 Seconds (390 Frames) |
| **Output Format** | MP4 (H.264 / yuv420p / CRF 17) |
| **Core Libraries** | NumPy, OpenCV (`cv2`), Multiprocessing, FFmpeg |
| **Rendering Method** | Procedural Mesh & Parametric Splines (No External 3D Models) |

---

## 📄 License

MIT License © [brocodesorg](https://github.com/brocodesorg) & [Avi Bhosale](https://github.com/AviBhosale01)
