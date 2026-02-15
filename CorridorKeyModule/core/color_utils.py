import torch
import numpy as np

def _is_tensor(x):
    return isinstance(x, torch.Tensor)

def to_srgb(x):
    """
    Converts Linear to sRGB.
    Supports both Numpy arrays and PyTorch tensors.
    """
    gamma = 1.0 / 2.2
    if _is_tensor(x):
        return torch.pow(x.clamp(min=0.0), gamma)
    else:
        return np.power(np.clip(x, 0.0, None), gamma)

def to_linear(x):
    """
    Converts sRGB to Linear.
    Supports both Numpy arrays and PyTorch tensors.
    """
    gamma = 2.2
    if _is_tensor(x):
        return torch.pow(x.clamp(min=0.0), gamma)
    else:
        return np.power(np.clip(x, 0.0, None), gamma)

def premultiply(fg, alpha):
    """
    Premultiplies foreground by alpha.
    fg: Color [..., C] or [C, ...]
    alpha: Alpha [..., 1] or [1, ...]
    """
    return fg * alpha

def unpremultiply(fg, alpha, eps=1e-6):
    """
    Un-premultiplies foreground by alpha.
    Ref: fg_straight = fg_premul / (alpha + eps)
    """
    if _is_tensor(fg):
        return fg / (alpha + eps)
    else:
        return fg / (alpha + eps)

def composite_straight(fg, bg, alpha):
    """
    Composites Straight FG over BG.
    Formula: FG * Alpha + BG * (1 - Alpha)
    """
    return fg * alpha + bg * (1.0 - alpha)

def composite_premul(fg, bg, alpha):
    """
    Composites Premultiplied FG over BG.
    Formula: FG + BG * (1 - Alpha)
    """
    return fg + bg * (1.0 - alpha)

def rgb_to_yuv(image):
    """
    Converts RGB to YUV (Rec. 601).
    Input: [..., 3, H, W] or [..., 3] depending on layout. 
    Supports standard PyTorch BCHW.
    """
    if not _is_tensor(image):
        raise TypeError("rgb_to_yuv only supports dict/tensor inputs currently")

    # Weights for RGB -> Y
    # Rec. 601: 0.299, 0.587, 0.114
    
    # Assume BCHW layout if 4 dims
    if image.dim() == 4:
        r = image[:, 0:1, :, :]
        g = image[:, 1:2, :, :]
        b = image[:, 2:3, :, :]
    elif image.dim() == 3 and image.shape[0] == 3: # CHW
        r = image[0:1, :, :]
        g = image[1:2, :, :]
        b = image[2:3, :, :]
    else:
        # Last dim conversion
        r = image[..., 0]
        g = image[..., 1]
        b = image[..., 2]

    y = 0.299 * r + 0.587 * g + 0.114 * b
    u = 0.492 * (b - y)
    v = 0.877 * (r - y)
    
    if image.dim() >= 3 and image.shape[-3] == 3: # Concatenate along Channel dim
         return torch.cat([y, u, v], dim=-3)
    else:
         return torch.stack([y, u, v], dim=-1)
