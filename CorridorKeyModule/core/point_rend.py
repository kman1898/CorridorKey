
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
PointRend Implementation for GreenFormer.
Focus: High-frequency edge refinement for Alpha and Foreground.
"""

def point_sample(input, point_coords, **kwargs):
    """
    Sample features from 'input' at 'point_coords'.
    input: [B, C, H, W]
    point_coords: [B, N, 2] in range [0, 1] OR [-1, 1] depending on align_corners.
                  We generally assume [0, 1] for logic, but grid_sample expects [-1, 1].
    """
    add_dim = False
    if point_coords.dim() == 3:
        add_dim = True
        point_coords = point_coords.unsqueeze(2) # [B, N, 1, 2]
    
    # Convert [0, 1] -> [-1, 1] for grid_sample
    # We assume standard normalized coordinates [0, 1] are passed
    grid = 2.0 * point_coords - 1.0
    
    # Flip Y if necessary? Usually grid_sample is (x, y). 
    # Let's assume point_coords are (x, y).
    
    # Can default match_corners to False if not provided
    if 'align_corners' not in kwargs:
        kwargs['align_corners'] = False
    
    output = F.grid_sample(input, grid, **kwargs) # [B, C, N, 1]
    
    if add_dim:
        output = output.squeeze(3) # [B, C, N]
        
    return output

def calculate_uncertainty(logits):
    """
    Estimate uncertainty of predictions.
    For logic (sigmoid): Uncertainty is minimized at 0 and 1, maximized at 0.5.
    Formula: - |p - 0.5|
    Or simply: 1.0 - |2p - 1|? 
    Standard PointRend uses top-k most uncertain.
    For Sigmoid P: Uncertainty = |P - 0.5|? No, we want low values for sure things.
    Let's use: (0.5 - |p - 0.5|) * 2 ?
       If p=0.5 -> 0.5 * 2 = 1.0 (Max uncertainty)
       If p=1.0 -> 0.0 (Min)
       If p=0.0 -> 0.0 (Min)
    """
    probs = torch.sigmoid(logits)
    uncertainty = 1.0 - torch.abs(probs - 0.5) * 2.0
    return uncertainty

def get_uncertain_point_coords_with_randomness(coarse_logits, uncertainty_func, num_points, oversample_ratio, beta):
    """
    Sample points based on uncertainty + randomness.
    
    1. Sample k * oversample_ratio points uniformly.
    2. Calculate uncertainty for these points.
    3. Select top k * beta most uncertain.
    4. Select remaining top k * (1 - beta) randomly?
    
    Actually standard PointRend logic:
    during training:
       1. Sample (oversample_ratio * N) points uniformly from [0, 1].
       2. Compute uncertainty map at these points (interpolate coarse prediction).
       3. Select top (beta * N) most uncertain points.
       4. Select (1 - beta) * N remaining points from the original random set (or fresh random).
    """
    assert oversample_ratio >= 1
    assert 0 <= beta <= 1
    
    B, _, H, W = coarse_logits.shape
    
    # 1. Generate random points
    num_sampled = int(num_points * oversample_ratio)
    point_coords = torch.rand(B, num_sampled, 2, device=coarse_logits.device)
    
    # 2. Get coarse features/predictions at these points
    point_logits = point_sample(coarse_logits, point_coords, align_corners=False) # [B, C, N]
    
    # 3. Compute uncertainty
    # We use Alpha channel (ch 0) for uncertainty metrics
    uncertainty = uncertainty_func(point_logits[:, 0:1, :]) # [B, 1, N]
    uncertainty = uncertainty.squeeze(1) # [B, N]
    
    # 4. Select Points
    num_uncertain = int(beta * num_points)
    num_random = num_points - num_uncertain
    
    # Top-k uncertain
    _, idx = uncertainty.topk(num_uncertain, dim=1) # [B, num_uncertain]
    
    shift = num_sampled * torch.arange(B, dtype=torch.long, device=coarse_logits.device)
    idx = idx + shift[:, None]
    
    point_coords_flat = point_coords.view(-1, 2)
    chosen_uncertain = point_coords_flat[idx.view(-1), :].view(B, num_uncertain, 2)
    
    # Random ones
    # We can just pick the first 'num_random' from the original random set (which are effectively random)
    # providing we shuffle? But torch.rand is random order anyway relative to image structure.
    # To be safe, let's just generate FRESH random points for the random portion.
    
    if num_random > 0:
        chosen_random = torch.rand(B, num_random, 2, device=coarse_logits.device)
        chosen_points = torch.cat([chosen_uncertain, chosen_random], dim=1)
    else:
        chosen_points = chosen_uncertain
        
    return chosen_points

class PointHead(nn.Module):
    def __init__(self, in_channels, out_channels=4, hidden_dim=256):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_channels)
        )
        
    def forward(self, x):
        # x: [B, C, N] -> [B, N, C]
        x = x.transpose(1, 2)
        x = self.mlp(x)
        return x.transpose(1, 2) # Back to [B, C, N]

class PointRendModule(nn.Module):
    def __init__(self, in_features, out_channels=4, num_points=4096):
        super().__init__()
        self.num_points = num_points
        self.point_head = PointHead(in_channels=in_features, out_channels=out_channels)
        
    def forward(self, fine_features, coarse_features, coarse_logits):
        """
        training_forward.
        fine_features: High-Res Image Features (The raw input image usually, or early conv features) [B, Cf, H, W]
        coarse_features: Backbone Features (UpSampled) [B, Cc, H, W] -- OPTIONAL, can essentially use coarse_logits as features
        coarse_logits: The output of the coarse decoder [B, 4, H, W]
        """
        
        # 1. Sample Points
        points = get_uncertain_point_coords_with_randomness(
            coarse_logits, 
            calculate_uncertainty, 
            num_points=self.num_points, 
            oversample_ratio=3, 
            beta=0.75
        )
        
        # 2. Extract Features at Points
        # A. Fine Features (e.g. from Image)
        fine_sampled = point_sample(fine_features, points, align_corners=False) # [B, Cf, N]
        
        # B. Coarse Features (e.g. from Coarse Prediction or Decoder Features)
        coarse_sampled = point_sample(coarse_features, points, align_corners=False) # [B, Cc, N]
        
        # Concatenate
        features = torch.cat([fine_sampled, coarse_sampled], dim=1) # [B, Cf+Cc, N]
        
        # 3. Predict
        point_logits = self.point_head(features) # [B, 4, N]
        
        return point_logits, points
