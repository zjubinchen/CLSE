import torch


def spatial_spectral_score_2d(x, t=1, h=24, w=24, cutoff_ratio=0.1):
    """
    Compute per-token high-frequency spectral scores via 2D FFT.
    Supports both image (t=1) and video (t>1) inputs by applying
    spatial 2D FFT independently per frame.

    Args:
        x:            token features [B, N, C], N = T*H*W
        t, h, w:      number of frames, height, width
        cutoff_ratio: Gaussian high-pass cutoff ratio
    Returns:
        score: [B, N]
    """
    B, N, C = x.shape
    # reshape to [B*T, C, H, W] for per-frame 2D FFT
    feat = x.transpose(1, 2).reshape(B, C, t, h, w)
    feat = feat.permute(0, 2, 1, 3, 4).reshape(B * t, C, h, w)

    fft_2d = torch.fft.fft2(feat.float())
    fft_shift = torch.fft.fftshift(fft_2d, dim=(-2, -1))

    # build Gaussian high-pass filter [H, W]
    center_h, center_w = h // 2, w // 2
    y, x_idx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    dist = torch.sqrt((y - center_h) ** 2 + (x_idx - center_w) ** 2).to(x.device)
    mask = 1 - torch.exp(-(dist ** 2) / (2 * (min(center_h, center_w) * cutoff_ratio) ** 2))
    if cutoff_ratio == 0.0:
        mask = torch.ones_like(mask)

    filtered = fft_shift * mask.unsqueeze(0).unsqueeze(0)
    filtered = torch.fft.ifftshift(filtered, dim=(-2, -1))
    high_freq = torch.fft.ifft2(filtered).abs()  # [B*T, C, H, W]

    # average over channels and flatten to [B, T*H*W]
    return high_freq.mean(dim=1).reshape(B, N)


def get_evolution_factor(s_L, s_Lk, temp=0.1, epsilon=1e-6):
    evolution_intensity = torch.abs(s_Lk - s_L) / (s_L.mean(dim=-1, keepdim=True) + s_L + epsilon)
    evolution_intensity = torch.clamp(evolution_intensity, max=1)
    mean_rate = evolution_intensity.mean(dim=-1, keepdim=True)
    std_rate = evolution_intensity.std(dim=-1, keepdim=True) + epsilon
    norm_rate = (evolution_intensity - mean_rate) / std_rate
    return torch.sigmoid(norm_rate / temp)


def calculate_evolution_score(z_L, z_Lk, attention_weights, image_grid_thw=(1, 24, 24),
                               cutoff=0.1, temp=0.1, score_type="clse_attn"):
    """
    Compute token importance scores for visual token pruning.

    Args:
        z_L:               image token features at layer L,   [B, N_img, C]
        z_Lk:              image token features at layer L+k, [B, N_img, C]
        attention_weights: text-to-image attention from the last text token
        image_grid_thw:    visual grid shape (T, H, W); T=1 for images
        cutoff:            spectral high-pass cutoff ratio
        temp:              normalization temperature
        score_type:        "attn" | "clse" | "clse_attn"

    Returns:
        final_score: [B, N_img]
    """
    if attention_weights is not None:
        if attention_weights.dim() == 4:
            attn_score = attention_weights.squeeze(2).mean(dim=1)
        elif attention_weights.dim() == 3:
            attn_score = attention_weights.squeeze(1)
        else:
            attn_score = attention_weights

    if score_type == "attn" or image_grid_thw is None:
        return attn_score


    t, h, w = image_grid_thw
    score_L  = spatial_spectral_score_2d(z_L,  t, h, w, cutoff_ratio=cutoff)
    score_Lk = spatial_spectral_score_2d(z_Lk, t, h, w, cutoff_ratio=cutoff)
    evo_factor = get_evolution_factor(score_L, score_Lk, temp=temp)

    if score_type == "clse_attn":
        return evo_factor * attn_score
    elif score_type == "clse":
        return evo_factor
    else:
        raise ValueError(f"Unknown score_type: {score_type}")
