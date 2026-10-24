# 🌹 Particle Rose — 3D Generative Particle Simulation & Live Animation

An extraordinarily beautiful, mathematically generated cinematic 3D rose simulation and real-time interactive graphics animation built with Python, NumPy, Pygame, and OpenCV.

The repository includes:
1. **`particle_rose.py`** — High-fidelity 139,340-particle 3D cinematic simulation with 48 layered petal whorls, 4 serrated leaves, 5 calyx sepals, optical bloom, depth rendering, continuous 3D camera orbits, and 1080×1920 (9:16 vertical) MP4 export.
2. **`main.py`** — Interactive live real-time Pygame animation rendering frame-by-frame on screen.
3. **`particle_rose.mp4`** — Pre-rendered 1080×1920 (9:16) 30 FPS showcase video (20.0 seconds, 600 frames).

---

## 📽️ Video Specifications

- **Format:** MP4 (`libx264` codec, `yuv420p`, CRF 14)
- **Resolution:** 1080 × 1920 (9:16 Vertical Portrait for Mobile / Shorts / Reels)
- **Frame Rate:** 30 FPS (600 frames total)
- **Duration:** 20.0 seconds
- **Particle Count:** ~139,340 active particles (petals, stem, 4 leaves, calyx sepals, thorns, atmospheric golden aura)

---

## 🎬 Cinematic Animation Phases (`particle_rose.py`)

1. **Scene 1: The Void Awakens (0.0s – 2.5s)**
   - Tiny ruby and golden glowing stardust embers emerge against pitch-black space (#000000).
   - Atmospheric particles float with gentle 3D turbulence.
2. **Scene 2: Harmonic Spiral Inflow & Stem Growth (2.5s – 6.5s)**
   - Particles accelerate into elegant logarithmic Fibonacci spirals, converging inward.
   - The emerald-green stem extends along a natural 3D Bezier curve with realistic thorns.
   - 5 calyx sepals emerge to cradle the base of the bud.
3. **Scene 3: Petals Blooming & Leaves Unfurling (3.5s – 10.5s)**
   - The central tight spiral bud coalesces first in deep ruby red.
   - Mid whorls cup outward and 4 serrated leaves expand along the stem.
   - Grand outer guard petals arch outward with soft rose-pink margins and shimmering golden rim highlights.
   - **By 10.5s, the entire rose is 100% completely bloomed and fully formed.**
4. **Scene 4: Grand 3D Orbit & Full Bloom Hold (10.5s – 20.0s — 9.5 Full Seconds!)**
   - The camera performs a buttery-smooth 3D orbital pan around the fully bloomed rose.
   - Thousands of golden ambient particles circulate gently around the flower.
   - Subtle organic breathing motion and shimmering highlights along the petal edges.
   - Finishes cleanly without abrupt cutoff.

---

## 🚀 Quick Start

### 1. Installation

Ensure you have Python 3.9+ and FFmpeg installed. Install dependencies:

```bash
pip install -r requirements.txt
```

### 2. Run the Live Interactive Graphics Window

```bash
python main.py
```

#### Interactive Controls:
| Key | Action |
| :--- | :--- |
| `SPACE` | Restart the bloom sequence |
| `P` | Pause / Resume animation |
| `M` | Toggle bloom / glow post-processing |
| `UP` / `DOWN` | Adjust animation playback speed |
| `ESC` | Exit application |

### 3. Generate Cinematic 1080×1920 MP4 Video

```bash
python particle_rose.py
```

---

## 🧮 Mathematical & Technical Highlights

- **3D Multi-tiered Rose Petal Surface Geometry:**
  Constructed across 48 parametric whorls with varying tilt, curvature, and wrap angles:
  $$r(\theta, \phi) = \rho_{base} + H_k \cdot \int \sin(\theta_{tilt}) \, ds + \Delta r_{ruffle}$$
- **Logarithmic Spiral Ingestion:**
  Particles gravitate from chaotic initial clouds to target flower manifold coordinates using smooth Hermite interpolation:
  $$S(t) = 3t^2 - 2t^3$$
- **Multi-pass Optical Bloom:**
  High-intensity emission buffers are blurred with multiple Gaussian kernels and composited via additive blending to recreate optical camera diffusion.
- **Dynamic 3D Camera Projection:**
  Full 3D rotation matrix transformations with perspective foreshortening and z-depth attenuation.

---

## 📦 Project Structure

```
particle-rose/
├── main.py              # Live interactive real-time Pygame animation
├── particle_rose.py     # 139,340-particle 3D cinematic video generator (20s, 600 frames)
├── particle_rose.mp4    # 1080x1920 (9:16) rendered showcase video (57 MB)
├── requirements.txt     # Python dependencies (pygame, numpy, opencv-python)
├── .gitignore           # Git ignore rules
└── README.md            # Documentation and instructions
```

---

## 📜 License

MIT License. Created with creative coding and mathematical art in Python.
