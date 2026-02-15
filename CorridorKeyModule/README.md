# CorridorKeyModule

A self-contained, high-performance AI Chroma Keying engine. This module provides a simple API to access the `CorridorKey` architecture (Hiera Backbone + CNN Refiner) for verifying and processing green screen footage.

## Features
*   **Resolution Independent:** Automatically resizes internal model embeddings to match your requested processing resolution (default 2048p).
*   **High Fidelity:** Preserves original input resolution using Lanczos4 resampling.
*   **Robust:** Handles Linear (EXR) and sRGB (PNG/MP4) inputs automatically.

## Installation

1. Copy the `CorridorKeyModule` folder to your project root.
2. Install dependencies:
   ```bash
   pip install -r CorridorKeyModule/requirements.txt
   ```
   *(Requires PyTorch, NumPy, OpenCV, Timm)*

## Usage

### 1. Initialization
Initialize the engine once. Point it to your `.pth` checkpoint. You can specify the internal processing resolution (e.g., `2048` for quality, `1024` for speed).

```python
from CorridorKeyModule import CorridorKeyEngine

# Initialize standard engine (CUDA)
# It will resize the checkpoint's Positional Embeddings to match 'img_size'
engine = CorridorKeyEngine(
    checkpoint_path="models/greenformer_v1.pth", 
    device='cuda', 
    img_size=2048
)
```

### 2. Processing a Frame
The engine expects:
*   **Image:** sRGB Numpy Array (H, W, 3). Range `0.0-1.0` (float) or `0-255` (uint8).
*   **Mask:** Linear Alpha Mask (H, W). Range `0.0-1.0` (float) or `0-255` (uint8).

```python
import cv2

# Load Image (sRGB)
img = cv2.imread("input.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Load Coarse Mask (Linear)
mask = cv2.imread("mask.png", cv2.IMREAD_GRAYSCALE)

# Process
# Returns dictionary keys: 'alpha', 'fg', 'comp'
result = engine.process_frame(img, mask)

# Save Results
# 'alpha' is Linear 0-1 float
# 'fg' is sRGB 0-1 float
alpha_uint8 = (result['alpha'] * 255).astype('uint8')
fg_uint8 = (result['fg'] * 255).astype('uint8')

cv2.imwrite("output_alpha.png", alpha_uint8)
cv2.imwrite("output_fg.png", cv2.cvtColor(fg_uint8, cv2.COLOR_RGB2BGR))
```

## Advanced

*   **Refiner Scale:** You can adjust the strength of the detail refiner at inference time (default 1.0).
    ```python
    result = engine.process_frame(img, mask, refiner_scale=1.5)
    ```

## Module Structure
*   `inference_engine.py`: Main API wrapper. Handles normalization, tensor conversion, and resizing.
*   `core/`: Internal model definitions (`model_transformer.py`), color logic (`color_utils.py`), and dependencies.
