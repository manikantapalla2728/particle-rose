# 🌹 Particle Rose — 3D Generative Particle Simulation & Live Animation

An extraordinarily beautiful, mathematically generated cinematic 3D rose simulation and real-time interactive graphics animation built with Python, NumPy, Pygame, and OpenCV.

The repository includes both:
1. **`particle_rose.py`** — High-fidelity 130,000-particle 3D cinematic simulation with optical bloom, depth rendering, 3D camera orbits, and 1080×1920 (9:16 vertical) MP4 export.
2. **`main.py`** — Interactive live real-time Pygame animation rendering frame-by-frame on screen.
3. **`particle_rose.mp4`** — Pre-rendered 1080×1920 (9:16) showcase video (24.0 seconds, 30 FPS).

---

## 📽️ Video Specifications

- **Format:** MP4 (`libx264` / `mp4v` codec)
- **Resolution:** 1080 × 1920 (9:16 Vertical Portrait for Mobile / Shorts / Reels)
- **Frame Rate:** 30 FPS (720 frames total)
- **Duration:** 24.0 seconds
- **Particle Count:** ~130,000 active particles (petals, stem, leaves, atmospheric dust)

---

## 🎬 Cinematic Animation Phases (`particle_rose.py`)

1. **Scene 1: Awakening (0.0s – 2.5s)**
   - Tiny ruby and golden glowing embers emerge from deep darkness (#000000).
   - Atmospheric particles float with gentle Perlin-like 3D turbulence.
2. **Scene 2: Spiral Particle Gathering (2.5s – 5.5s)**
   - Particles trace logarithmic spiral trajectories, coalescing towards mathematical attractors.
   - Core glow intensifies with additive alpha blending.
3. **Scene 3: 3D Petal Bloom (5.5s – 10.0s)**
   - Parametric 3D rose petal surfaces unfold in multi-tiered whorls.
   - Gradients span deep crimson, ruby red, vibrant vermilion, and soft rose-gold edge highlights.
4. **Scene 4: Stem & Leaf Emergence (7.0s – 11.0s)**
   - Emerald and jade-green stem spirals downward.
   - Two serrated, pointed leaves emerge with realistic vein distribution.
5. **Scene 5: Extended Full-Bloom Orbit & Radiance (11.0s – 24.0s)**
   - The rose is 100% formed and held in its full blooming glory.
   - Front three-quarter cinematic camera with gentle 3D parallax orbit ($\pm 15^\circ$) showcasing the petals' depth without turning away.
   - Multi-pass optical bloom, shimmering golden ambient dust circulation, and a gentle cinematic fade.

---

## 🚀 Quick Start

### 1. Installation

Ensure you have Python 3.9+ installed. Install the dependencies:

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

This will run the full 130k-particle 3D simulation with progress output and generate `particle_rose.mp4` (1080×1920, 9:16).

---

## 🧮 Mathematical & Technical Highlights

- **3D Rose Petal Surface Geometry:**
  Constructed using parametric polar and spherical coordinates:
  $$r(\theta, \phi) = a \cdot (1 - \cos(k \theta)) \cdot \sin(\phi)$$
- **Logarithmic Spiral Ingestion:**
  Particles gravitate from chaotic initial states to target flower manifold coords using smooth Hermite interpolation:
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
├── particle_rose.py     # 130,000-particle 3D cinematic video generator
├── particle_rose.mp4    # 1080x1920 (9:16) rendered showcase video
├── requirements.txt     # Python dependencies (pygame, numpy, opencv-python)
├── .gitignore           # Git ignore rules
└── README.md            # Documentation and instructions
```

---

## 📜 License

MIT License. Created with creative coding and mathematical art in Python.
