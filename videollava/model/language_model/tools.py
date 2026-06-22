import torch


def spatial_spectral_score_per_frame(x, t=8, h=16, w=16, cutoff_ratio=0.1):
    """
    Compute per-token high-frequency spectral scores via per-frame 2D FFT.

    For sparse video (e.g. 8 frames), 3D FFT suffers from temporal aliasing due to
    insufficient temporal sampling. This function merges B and T into the batch dimension
    and applies 2D FFT independently per frame, capturing spatial high-frequency structure
    while relying on cross-layer evolution to capture temporal change.

    Args:
        x:            token features [B, N, C], N = t * h * w
        t, h, w:      temporal and spatial dimensions of the visual token grid
        cutoff_ratio: Gaussian high-pass cutoff ratio
    Returns:
        score: [B, N]
    """
    B, N, C = x.shape
    device = x.device

    # reshape: [B, T*H*W, C] -> [B, T, H, W, C] -> [B*T, C, H, W]
    feat_frames = x.reshape(B, t, h, w, C).permute(0, 1, 4, 2, 3).reshape(B * t, C, h, w)

    # 2D FFT over spatial dims for all B*T frames in parallel
    fft_2d = torch.fft.fft2(feat_frames.float(), dim=(-2, -1))
    fft_shift = torch.fft.fftshift(fft_2d, dim=(-2, -1))

    # build Gaussian high-pass filter [H, W]
    center_h, center_w = h // 2, w // 2
    y_idx, x_idx = torch.meshgrid(
        torch.arange(h, device=device),
        torch.arange(w, device=device),
        indexing='ij'
    )
    dist_sq = (y_idx - center_h) ** 2 + (x_idx - center_w) ** 2
    sigma = (min(h, w) // 2) * cutoff_ratio
    mask = 1 - torch.exp(-dist_sq / (2 * sigma ** 2 + 1e-6))

    filtered = fft_shift * mask.unsqueeze(0).unsqueeze(0)
    filtered = torch.fft.ifftshift(filtered, dim=(-2, -1))
    high_freq = torch.fft.ifft2(filtered, dim=(-2, -1)).abs()

    # average over channels and flatten to [B, N]
    return high_freq.mean(dim=1).reshape(B, N)


def spatial_spectral_score_3d(x, t=8, h=16, w=16, cutoff_ratio=0.1):
    """
    Compute per-token high-frequency spectral scores via 3D FFT.

    Args:
        x:            token features [B, N, C], N = t * h * w
        t, h, w:      temporal and spatial dimensions of the visual token grid
        cutoff_ratio: Gaussian high-pass cutoff ratio
    Returns:
        score: [B, N]
    """
    B, N, C = x.shape
    device = x.device

    # reshape to [B, C, T, H, W] for 3D FFT
    feat_cube = x.transpose(1, 2).reshape(B, C, t, h, w)

    fft_3d = torch.fft.fftn(feat_cube.float(), dim=(-3, -2, -1))
    fft_shift = torch.fft.fftshift(fft_3d, dim=(-3, -2, -1))

    # build Gaussian high-pass filter [T, H, W]
    center_t, center_h, center_w = t // 2, h // 2, w // 2
    z_idx, y_idx, x_idx = torch.meshgrid(
        torch.arange(t, device=device),
        torch.arange(h, device=device),
        torch.arange(w, device=device),
        indexing='ij'
    )
    dist_sq = (z_idx - center_t) ** 2 + (y_idx - center_h) ** 2 + (x_idx - center_w) ** 2
    sigma = (min(t, h, w) // 2) * cutoff_ratio
    mask = 1 - torch.exp(-dist_sq / (2 * sigma ** 2))

    filtered = fft_shift * mask.unsqueeze(0).unsqueeze(0)
    filtered = torch.fft.ifftshift(filtered, dim=(-3, -2, -1))
    high_freq = torch.fft.ifftn(filtered, dim=(-3, -2, -1)).abs()

    # average over channels and flatten to [B, N]
    return high_freq.mean(dim=1).reshape(B, N)


def get_evolution_factor(s_L, s_Lk, temp=0.1, epsilon=1e-6):
    evolution_intensity = torch.abs(s_Lk - s_L) / (s_L.mean(dim=-1, keepdim=True) + s_L + epsilon)
    evolution_intensity = torch.clamp(evolution_intensity, max=1)
    mean_rate = evolution_intensity.mean(dim=-1, keepdim=True)
    std_rate = evolution_intensity.std(dim=-1, keepdim=True) + epsilon
    norm_rate = (evolution_intensity - mean_rate) / std_rate
    return torch.sigmoid(norm_rate / temp)



def calculate_evolution_score(z_L, z_Lk, attention_weights, image_grid_thw=(8, 16, 16), score_type="clse_attn"):
    """
    Compute token importance scores for visual token pruning.

    Args:
        z_L:               image token features at layer L,   [B, N_img, C]
        z_Lk:              image token features at layer L+k, [B, N_img, C]
        attention_weights: text-to-image attention from the last text token
        image_grid_thw:    visual grid shape (T, H, W); None falls back to attention-only scoring
        cutoff:            spectral high-pass cutoff ratio
        temp:              normalization temperature
        score_type:        "attn" | "clse" | "clse_attn"

    Returns:
        final_score: [B, N_img]
    """
    if attention_weights.dim() == 4:
        attn_score = attention_weights.squeeze(2).mean(dim=1)
    elif attention_weights.dim() == 3:
        attn_score = attention_weights.squeeze(1)
    else:
        attn_score = attention_weights

    if score_type == "attn" or image_grid_thw is None:
        return attn_score

    t, h, w = image_grid_thw
    score_L  = spatial_spectral_score_per_frame(z_L,  t, h, w)
    score_Lk = spatial_spectral_score_per_frame(z_Lk, t, h, w)
    evo_factor = get_evolution_factor(score_L, score_Lk)

    if score_type == "clse_attn":
        return evo_factor * attn_score
    elif score_type == "clse":
        return evo_factor
    else:
        raise ValueError(f"Unknown score_type: {score_type}")
