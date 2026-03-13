from timm.layers.drop import drop_path
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import DropPath, trunc_normal_, lecun_normal_
import math
from functools import partial
from mamba_ssm import Mamba  # Giả sử bạn đã import từ baseline gốc


class TemporalPyramidPooling(nn.Module):
    """
    Multi-scale temporal pyramid pooling để capture different temporal resolutions.
    Sử dụng AdaptiveAvgPool1d với multiple outputs rồi concat và project.
    """
    def __init__(self, dim, pool_sizes=[1, 2, 4, 8], reduction_ratio=4):
        super().__init__()
        self.dim = dim
        self.pool_sizes = pool_sizes
        self.pools = nn.ModuleList([
            nn.AdaptiveAvgPool1d(size) for size in pool_sizes
        ])
        
        # Project mỗi scale về dim_reduced rồi concat
        self.dim_reduced = dim // reduction_ratio
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(dim, self.dim_reduced, kernel_size=1, bias=False),
                nn.BatchNorm1d(self.dim_reduced),
                nn.ReLU(inplace=True)
            ) for _ in pool_sizes
        ])
        
        # Final projection để về lại dim gốc
        total_dim = self.dim_reduced * len(pool_sizes)
        self.final_proj = nn.Sequential(
            nn.Conv1d(total_dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True)
        )
        
        # Residual connection weight
        self.alpha = nn.Parameter(torch.tensor(0.1))  # Start small
    
    def forward(self, x):
        """
        x: [B, T, C] input
        return: [B, T, C] enhanced with multi-scale info
        """
        B, T, C = x.shape
        x_conv = x.transpose(1, 2)  # [B, C, T] for conv operations
        
        # Multi-scale pooling
        pyramid_features = []
        for pool, proj in zip(self.pools, self.projections):
            # Pool to different scales
            pooled = pool(x_conv)  # [B, C, pool_size]
            # Project to reduced dim
            projected = proj(pooled)  # [B, dim_reduced, pool_size]
            # Upsample back to original temporal size
            upsampled = F.interpolate(projected, size=T, mode='linear', align_corners=False)
            pyramid_features.append(upsampled)
        
        # Concatenate all scales
        pyramid_concat = torch.cat(pyramid_features, dim=1)  # [B, total_dim, T]
        
        # Final projection
        pyramid_out = self.final_proj(pyramid_concat)  # [B, C, T]
        
        # Residual connection
        enhanced = x_conv + self.alpha * pyramid_out
        return enhanced.transpose(1, 2)  # [B, T, C]


class SEModule(nn.Module):
    """
    Squeeze-and-Excitation module for channel-wise recalibration based on global context.
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        x: [B, T, C]
        return: [B, T, C] with channel recalibration
        """
        B, T, C = x.shape
        # Global average pooling over temporal dimension
        y = x.transpose(1, 2)  # [B, C, T]
        y = self.avg_pool(y)  # [B, C, 1]
        y = y.view(B, C)  # [B, C]
        
        # Generate channel weights
        y = self.fc(y).view(B, 1, C)  # [B, 1, C]
        
        # Scale channels
        return x * y


class BidirectionalMamba(nn.Module):
    """
    Bidirectional Mamba layer: processes sequence in both forward and backward directions.
    """
    def __init__(self, dim, d_state=64, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        
        # Forward and backward Mamba blocks
        self.mamba_forward = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        self.mamba_backward = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        
        # Fusion gate for combining forward and backward
        self.fusion_gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )
        
        # Output projection
        self.out_proj = nn.Linear(dim * 2, dim)
    
    def forward(self, x):
        """
        x: [B, T, C]
        return: [B, T, C] bidirectional processed
        """
        x_norm = self.norm(x)
        
        # Forward pass
        x_forward = self.mamba_forward(x_norm)
        
        # Backward pass (reverse sequence)
        x_reversed = torch.flip(x_norm, dims=[1])
        x_backward = self.mamba_backward(x_reversed)
        x_backward = torch.flip(x_backward, dims=[1])
        
        # Concatenate and gate
        x_concat = torch.cat([x_forward, x_backward], dim=-1)  # [B, T, 2*C]
        gate = self.fusion_gate(x_concat)  # [B, T, C]
        
        # Weighted combination
        x_out = self.out_proj(x_concat)  # [B, T, C]
        
        return x + x_out  # Residual connection


class GlobalMHSA(nn.Module):
    """
    Global Multi-Head Self-Attention with Relative Position Encoding.
    Used in the final stage for capturing long-range dependencies.
    """
    def __init__(self, dim, heads=8, attn_drop=0.0, proj_drop=0.0, max_seq_len=4096):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.heads = heads
        self.dim_head = dim // heads
        self.scale = self.dim_head ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # Relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * max_seq_len - 1), heads)
        )
        trunc_normal_(self.relative_position_bias_table, std=0.02)
        
    def _get_relative_position_bias(self, seq_len):
        """Generate relative position bias for given sequence length."""
        # Create position indices
        coords = torch.arange(seq_len)
        relative_coords = coords[:, None] - coords[None, :]  # [seq_len, seq_len]
        relative_coords += seq_len - 1  # Shift to positive indices
        
        # Clamp to table size
        max_idx = self.relative_position_bias_table.size(0) - 1
        relative_coords = relative_coords.clamp(0, max_idx)
        
        # Gather bias values
        relative_bias = self.relative_position_bias_table[relative_coords]  # [seq_len, seq_len, heads]
        return relative_bias.permute(2, 0, 1).unsqueeze(0)  # [1, heads, seq_len, seq_len]
    
    def forward(self, x):
        """
        x: [B, T, C]
        return: [B, T, C] with global attention
        """
        B, T, C = x.shape
        x_norm = self.norm(x)
        
        # Generate Q, K, V
        qkv = self.qkv(x_norm).reshape(B, T, 3, self.heads, self.dim_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, heads, T, dim_head]
        
        # Scaled dot-product attention with relative position bias
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # Add relative position bias
        rel_pos_bias = self._get_relative_position_bias(T).to(x.device)
        attn = attn + rel_pos_bias
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Apply attention to values
        x_attn = (attn @ v).transpose(1, 2).reshape(B, T, C)
        x_attn = self.proj(x_attn)
        x_attn = self.proj_drop(x_attn)
        
        return x + x_attn


class LateralConnection(nn.Module):
    """
    Kết nối nhanh-chậm (fast <-> slow) với căn chỉnh thời gian an toàn và nhiều kiểu fusion.
    """
    def __init__(
        self,
        fast_channels=36,
        slow_channels=72,
        reverse=False,
        fusion_type='additive_gated',
        time_scale=2
    ):
        super().__init__()
        self.reverse = reverse
        self.fusion_type = fusion_type
        self.time_scale = time_scale

        if not reverse:
            # Fast -> Slow
            self.align_proj = nn.Sequential(
                nn.Conv1d(fast_channels, slow_channels, kernel_size=3, stride=time_scale, padding=1, bias=False),
                nn.BatchNorm1d(slow_channels),
                nn.ReLU(inplace=True)
            )
            target_channels = slow_channels
        else:
            # Slow -> Fast
            self.align_proj = nn.Sequential(
                nn.Upsample(scale_factor=time_scale, mode='linear', align_corners=False),
                nn.Conv1d(slow_channels, fast_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm1d(fast_channels),
                nn.ReLU(inplace=True)
            )
            target_channels = fast_channels

        if fusion_type in ['additive_gated', 'film']:
            self.gate = nn.Sequential(
                nn.Conv1d(target_channels * 2, target_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(target_channels),
                nn.SiLU(),
                nn.Conv1d(target_channels, target_channels, kernel_size=1),
                nn.Sigmoid()
            )
        
        if fusion_type == 'film':
            self.shift = nn.Sequential(
                nn.Conv1d(target_channels * 2, target_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(target_channels),
                nn.SiLU(),
                nn.Conv1d(target_channels, target_channels, kernel_size=1)
            )

        self.alpha = nn.Parameter(torch.tensor(1.0))

    @staticmethod
    def _pad_or_crop(x_c_first, T_target):
        B, C, T = x_c_first.shape
        if T == T_target:
            return x_c_first
        if T > T_target:
            return x_c_first[:, :, :T_target]
        pad_T = T_target - T
        return F.pad(x_c_first, (0, pad_T))

    def forward(self, target, source):
        source_c_first = source.permute(0, 2, 1)
        aligned = self.align_proj(source_c_first)
        T_target = target.size(1)
        aligned = self._pad_or_crop(aligned, T_target)
        aligned_btC = aligned.permute(0, 2, 1)

        if self.fusion_type == 'additive_gated':
            tgt_c_first = target.permute(0, 2, 1)
            gate_in = torch.cat([tgt_c_first, aligned], dim=1)
            g = self.gate(gate_in).permute(0, 2, 1)
            out = target + self.alpha * (g * aligned_btC)
            return out

        elif self.fusion_type == 'multiplicative':
            out = target + self.alpha * (target * torch.sigmoid(aligned_btC))
            return out

        elif self.fusion_type == 'film':
            tgt_c_first = target.permute(0, 2, 1)
            film_in = torch.cat([tgt_c_first, aligned], dim=1)
            gamma = self.gate(film_in).permute(0, 2, 1)
            beta = self.shift(film_in).permute(0, 2, 1)
            out = target + self.alpha * (gamma * aligned_btC + beta)
            return out

        else:  # 'residual'
            return target + self.alpha * aligned_btC


class WindowMHSA1D(nn.Module):
    """
    Local MHSA 1D với overlap-add chuẩn và SE module.
    """
    def __init__(
        self,
        dim,
        heads=4,
        attn_drop=0.0,
        proj_drop=0.0,
        window_size=128,
        use_shifted=True,
        hann_weight=True,
        chunk_size=2048,
        use_se=True  # Add SE module option
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads=heads, dropout=attn_drop, batch_first=True)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(proj_drop)
        
        # Add SE module
        self.use_se = use_se
        if use_se:
            self.se = SEModule(dim)

        self.window_size = window_size
        self.pad_w = (window_size - 1) // 2
        self.use_shifted = use_shifted
        self.hann_weight = hann_weight
        self.chunk_size = chunk_size

        if hann_weight:
            win = torch.hann_window(window_size, periodic=False)
            self.register_buffer("hann", win, persistent=False)
        else:
            self.hann = None

    @torch.no_grad()
    def _overlap_count(self, B, L, device, C=1):
        ones = torch.ones(B, C * self.window_size, L, device=device)
        counts = F.fold(
            ones,
            output_size=(1, L),
            kernel_size=(1, self.window_size),
            stride=(1, 1),
            padding=(0, self.pad_w),
        )
        return counts

    def _attend_windows(self, x):
        B, T, C = x.shape
        y = self.norm(x)

        y2d = y.permute(0, 2, 1).unsqueeze(2)
        patches = F.unfold(
            y2d,
            kernel_size=(1, self.window_size),
            stride=(1, 1),
            padding=(0, self.pad_w),
        )

        B_, CW, L = patches.shape
        w = self.window_size
        assert CW == C * w
        windows = rearrange(patches, 'b (c w) l -> (b l) w c', c=C, w=w)

        if self.hann is not None:
            windows = windows * self.hann.to(windows.device).view(1, w, 1)

        N = windows.size(0)
        outs = []
        for i in range(0, N, self.chunk_size):
            chunk = windows[i:i + self.chunk_size]
            out, _ = self.attn(chunk, chunk, chunk, need_weights=False)
            outs.append(out)
        windows_out = torch.cat(outs, dim=0)

        patches_out = rearrange(windows_out, '(b l) w c -> b (c w) l', b=B, l=L)

        y2d_out = F.fold(
            patches_out,
            output_size=(1, T),
            kernel_size=(1, w),
            stride=(1, 1),
            padding=(0, self.pad_w),
        )

        counts = self._overlap_count(B, T, x.device, C=1)
        y2d_out = y2d_out / (counts + 1e-6)

        y_out = y2d_out.squeeze(2).permute(0, 2, 1)
        return y_out

    def forward(self, x):
        B, T, C = x.shape
        if T <= self.window_size:
            y, _ = self.attn(self.norm(x), self.norm(x), self.norm(x), need_weights=False)
            y = self.drop(self.proj(y))
            if self.use_se:
                y = self.se(y)
            return x + y

        y_main = self._attend_windows(x)

        if self.use_shifted:
            shift = self.window_size // 2
            x_shift = torch.roll(x, shifts=shift, dims=1)
            y_shift = self._attend_windows(x_shift)
            y_shift = torch.roll(y_shift, shifts=-shift, dims=1)
            y = 0.5 * (y_main + y_shift)
        else:
            y = y_main

        y = self.drop(self.proj(y))
        
        # Apply SE module for channel recalibration
        if self.use_se:
            y = self.se(y)
        
        return x + y


class ConvModule1D(nn.Module):
    def __init__(self, dim, drop=0.1):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.pw1 = nn.Conv1d(dim, 2*dim, kernel_size=1)
        self.dw = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.bn = nn.BatchNorm1d(dim)
        self.act = nn.SiLU()
        self.pw2 = nn.Conv1d(dim, dim, kernel_size=1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        y = self.ln(x).transpose(1, 2)
        y = self.pw1(y)
        a, b = y.chunk(2, dim=1)
        y = a * torch.sigmoid(b)
        y = self.dw(y)
        y = self.bn(y)
        y = self.act(y)
        y = self.pw2(y)
        y = self.drop(y).transpose(1, 2)
        return x + y


class FFN1D(nn.Module):
    def __init__(self, dim, mult=4, drop=0.1, residual_scale=0.5):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * mult)
        self.fc2 = nn.Linear(dim * mult, dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop)
        self.scale = residual_scale
    
    def forward(self, x):
        y = self.norm(x)
        y = self.fc2(self.drop(self.act(self.fc1(y))))
        return x + self.scale * y


class ConFormerBlock1D(nn.Module):
    """
    ConFormer block with SE module for channel recalibration.
    """
    def __init__(self, dim, heads=4, ffn_mult=4, drop=0.1, drop_path=0.0, use_se=True):
        super().__init__()
        self.ffn1 = FFN1D(dim, mult=ffn_mult, drop=drop, residual_scale=0.5)
        self.mhsa = WindowMHSA1D(dim, heads=heads, attn_drop=drop, proj_drop=drop, use_se=use_se)
        self.conv = ConvModule1D(dim, drop=drop)
        self.ffn2 = FFN1D(dim, mult=ffn_mult, drop=drop, residual_scale=0.5)
        self.dp = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        
        # Add SE module for global context
        self.use_se = use_se
        if use_se:
            self.se = SEModule(dim, reduction=16)
    
    def forward(self, x):
        x = x + self.dp(self.ffn1(x))
        x = x + self.dp(self.mhsa(x))
        x = x + self.dp(self.conv(x))
        x = x + self.dp(self.ffn2(x))
        
        # Apply SE for global channel recalibration
        if self.use_se:
            x = self.se(x)
        
        return x


class MultiDWConvBlock1D(nn.Module):
    def __init__(self, c, ks=[3,5,7], ds=[1,3,5], drop=0.0, drop_path=0.0):
        super().__init__()
        assert len(ks) == len(ds)
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(c, c, kernel_size=k, padding=d*(k//2), dilation=d, groups=c, bias=False),
                nn.Conv1d(c, c, kernel_size=1, bias=False),
                nn.BatchNorm1d(c),
                nn.GELU()
            ) for k, d in zip(ks, ds)
        ])
        self.proj = nn.Conv1d(c * len(ks), c, kernel_size=1)
        self.drop = nn.Dropout(drop)
        self.dp = DropPath(drop_path) if drop_path > 0 else nn.Identity()
    
    def forward(self, x):
        y = x.transpose(1, 2)
        ys = [branch(y) for branch in self.branches]
        y = torch.cat(ys, dim=1)
        y = self.proj(y)
        y = self.drop(y).transpose(1, 2)
        return x + self.dp(y)


class FastPathBlock(nn.Module):
    def __init__(self, dim, drop=0.1, drop_path=0.0):
        super().__init__()
        self.block = MultiDWConvBlock1D(dim, drop=drop, drop_path=drop_path)
    
    def forward(self, x):
        return self.block(x)


class HybridConFormerSF(nn.Module):
    """
    Enhanced Hybrid ConFormer-SlowFast with:
    - Global attention in final stage
    - Bidirectional Mamba
    - SE modules for channel recalibration
    """
    def __init__(self, dim, heads=4, drop=0.1):
        super().__init__()
        self.dim = dim
        self.Stem_slow = nn.Sequential(
            nn.Conv1d(dim, dim * 2, 3, 2, 1),
            nn.BatchNorm1d(dim*2),
            nn.ReLU(inplace=True)
        )

        dpr = torch.linspace(0, 0.2, steps=4).tolist()

        # Stage 1-2: ConFormer with SE modules
        self.block1 = nn.ModuleList([
            ConFormerBlock1D(dim=dim*2, heads=heads, ffn_mult=4, drop=drop, drop_path=dpr[i], use_se=True)
            for i in range(4)
        ])
        self.block2 = nn.ModuleList([
            ConFormerBlock1D(dim=dim*2, heads=heads, ffn_mult=4, drop=drop, drop_path=dpr[i], use_se=True)
            for i in range(4)
        ])
        
        # Stage 3: Hybrid with Bidirectional Mamba and Global Attention
        self.block3 = nn.ModuleList()
        for i in range(4):
            if i < 2:
                # First 2 blocks: Bidirectional Mamba
                self.block3.append(nn.Sequential(
                    nn.LayerNorm(dim*2),
                    BidirectionalMamba(dim*2)
                ))
            else:
                # Last 2 blocks: Global attention with relative PE
                self.block3.append(GlobalMHSA(dim*2, heads=heads, attn_drop=drop, proj_drop=drop))

        # Fast path blocks
        self.block1_fast = nn.ModuleList([
            FastPathBlock(dim=dim, drop=drop, drop_path=dpr[i])
            for i in range(4)
        ])
        self.block2_fast = nn.ModuleList([
            FastPathBlock(dim=dim, drop=drop, drop_path=dpr[i])
            for i in range(4)
        ])
        self.block3_fast = nn.ModuleList([
            FastPathBlock(dim=dim, drop=drop, drop_path=dpr[i])
            for i in range(4)
        ])
        
        # Multi-scale pyramid pooling
        self.pyramid_pool_1 = TemporalPyramidPooling(dim, pool_sizes=[1, 2, 4, 8])
        self.pyramid_pool_2 = TemporalPyramidPooling(dim, pool_sizes=[1, 2, 4, 8])
        self.pyramid_pool_3 = TemporalPyramidPooling(dim, pool_sizes=[1, 2, 4, 8])

        # Bidirectional lateral connections
        self.fuse_1 = LateralConnection(fast_channels=dim, slow_channels=dim*2, fusion_type='additive_gated')
        self.fuse_1_rev = LateralConnection(fast_channels=dim, slow_channels=dim*2, reverse=True, fusion_type='additive_gated')
        self.fuse_2 = LateralConnection(fast_channels=dim, slow_channels=dim*2, fusion_type='additive_gated')
        self.fuse_2_rev = LateralConnection(fast_channels=dim, slow_channels=dim*2, reverse=True, fusion_type='additive_gated')
        self.fuse_3 = LateralConnection(fast_channels=dim, slow_channels=dim*2, fusion_type='additive_gated')
        self.fuse_3_rev = LateralConnection(fast_channels=dim, slow_channels=dim*2, reverse=True, fusion_type='additive_gated')

        self.upsample = nn.Upsample(scale_factor=2, mode='linear', align_corners=False)
        
        # Global SE for final fusion
        self.global_se = SEModule(dim * 3)
    
    def forward(self, x):
        temp_fast = x.permute(0, 2, 1)
        pe_fast = get_sinusoidal_pe(temp_fast.size(1), self.dim, x.device)
        x_fast = temp_fast + pe_fast

        x_slow = self.Stem_slow(x)
        temp_slow = x_slow.permute(0, 2, 1)
        pe_slow = get_sinusoidal_pe(temp_slow.size(1), self.dim * 2, x.device)
        x_slow = temp_slow + pe_slow

        # Stage 1
        for blk in self.block1: x_slow = blk(x_slow)
        for blk in self.block1_fast: x_fast = blk(x_fast)
        x_fast = self.pyramid_pool_1(x_fast)
        
        x_slow = self.fuse_1(x_slow, x_fast)
        x_fast = self.fuse_1_rev(x_fast, x_slow)

        # Stage 2
        for blk in self.block2: x_slow = blk(x_slow)
        for blk in self.block2_fast: x_fast = blk(x_fast)
        x_fast = self.pyramid_pool_2(x_fast)
        
        x_slow = self.fuse_2(x_slow, x_fast)
        x_fast = self.fuse_2_rev(x_fast, x_slow)

        # Stage 3 (with hybrid Mamba and global attention)
        for blk in self.block3: x_slow = blk(x_slow)
        for blk in self.block3_fast: x_fast = blk(x_fast)
        x_fast = self.pyramid_pool_3(x_fast)
        
        x_slow = self.fuse_3(x_slow, x_fast)
        x_fast = self.fuse_3_rev(x_fast, x_slow)

        # Upsample and match T
        x_slow = x_slow.permute(0, 2, 1)
        x_slow = self.upsample(x_slow)
        x_slow = x_slow.permute(0, 2, 1)

        T_fast = x_fast.size(1)
        T_slow = x_slow.size(1)
        if T_slow > T_fast:
            x_slow = x_slow[:, :T_fast, :]
        elif T_slow < T_fast:
            x_slow = F.pad(x_slow, (0, 0, 0, T_fast - T_slow))

        x_fusion = torch.cat((x_slow, x_fast), dim=2)
        
        # Apply global SE for final feature recalibration
        x_fusion = self.global_se(x_fusion)
        
        return x_fusion


def get_sinusoidal_pe(T, C, device):
    """Generate sinusoidal positional encoding."""
    position = torch.arange(0, T, dtype=torch.float).unsqueeze(1).to(device)
    div_term = torch.exp(torch.arange(0, C, 2).float() * (-torch.log(torch.tensor(10000.0)) / C)).to(device)
    pe = torch.zeros(T, C).to(device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0)
    return pe


class SlowFast(nn.Module):
    """
    Enhanced SlowFast model with:
    - Hybrid ConFormer-Mamba architecture
    - Global attention in final stages
    - Bidirectional Mamba layers
    - SE modules for channel-wise recalibration
    - Multi-scale temporal pyramid pooling
    """
    def __init__(self, out_channels=3):
        super(SlowFast, self).__init__()
        dim = 128
        self.Stem = nn.Sequential(
            nn.Conv1d(36, dim, 3, 1, 1),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(dim, dim, 3, 1, 1),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True)
        )

        self.spot_pathway = HybridConFormerSF(dim)
        self.recog_pathway = HybridConFormerSF(dim)
        
        # Enhanced heads with SE
        self.fc_spot = nn.Sequential(
            SEModule(dim*3, reduction=8),
            nn.Linear(in_features=dim*3, out_features=1)
        )
        
        self.fc_recog = nn.Sequential(
            SEModule(dim*3, reduction=8),
            nn.Linear(in_features=dim*3, out_features=out_channels)
        )
        
        self.sigmoid = nn.Sigmoid()

        # Initialize weights
        self.apply(segm_init_weights)
        self.apply(partial(_init_weights, n_layer=3, **{}))

    def forward(self, x):
        x = x.squeeze(1)
        x = self.Stem(x)

        # Spot pathway
        x_spot = self.spot_pathway(x)
        x_spot_prob = self.sigmoid(self.fc_spot(x_spot)).squeeze(-1)  # [B, T]

        # Recognition pathway
        x_recog = self.recog_pathway(x)
        x_recog = self.fc_recog(x_recog)  # [B, T, out_channels]

        # Mask recog by spot (spot-then-recognize)
        x_recog = x_recog * x_spot_prob.unsqueeze(-1)

        return x_spot_prob, x_recog


def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,
):
    """Initialize model weights with GPT-2 style initialization."""
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


def segm_init_weights(m):
    """Initialize segmentation model weights."""
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv1d):
        lecun_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm1d)):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)