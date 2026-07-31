"""Model architectures.

LUCIDNet is the paper's architecture (TextEncoder + SpatialEncoder + CrossAttention +
conditioned U-Net), carried over from the original notebook with the dead ``mask``
forward-argument removed (conditioning already flows in through ``spatial_embed``,
which the caller computes from the mask — see ANALYSIS.md bug #11).

AttentionUNet and TimmEncoderUNet are two additional baselines added for a fair
ablation / to try to beat the paper's numbers with stronger, transfer-learned features
(the biggest lever available on 35-80 image training sets).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
except ImportError:  # pragma: no cover
    timm = None


# --------------------------------------------------------------------------- #
# Shared conditioning modules (unchanged from the original notebook)
# --------------------------------------------------------------------------- #

class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_heads=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.final_fc = nn.Linear(embed_dim, embed_dim)

    def forward(self, text_tokens):
        embedded = self.embedding(text_tokens)
        encoded = self.transformer(embedded)
        pooled = torch.mean(encoded, dim=1)
        return self.final_fc(pooled)


class SpatialEncoder(nn.Module):
    def __init__(self, in_channels=1, base_channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(base_channels, base_channels * 2, 3, padding=1, stride=2)
        self.conv3 = nn.Conv2d(base_channels * 2, base_channels * 4, 3, padding=1, stride=2)
        self.conv4 = nn.Conv2d(base_channels * 4, base_channels * 8, 3, padding=1, stride=2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        x3 = F.relu(self.conv3(x2))
        x4 = F.relu(self.conv4(x3))
        spatial_feat = self.adaptive_pool(x4)
        return spatial_feat.squeeze(-1).squeeze(-1), [x1, x2, x3, x4]


class CrossAttention(nn.Module):
    def __init__(self, channels, cond_dim, num_heads=4):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.cond_proj = nn.Linear(cond_dim, channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x, cond_embed):
        B, C, H, W = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        x_flat = self.norm(x_flat)
        cond = self.cond_proj(cond_embed).unsqueeze(1)
        attn_out, _ = self.attn(query=x_flat, key=cond, value=cond)
        attn_out = attn_out.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return x + attn_out


# --------------------------------------------------------------------------- #
# Model 1: LUCID-Net (paper architecture, dead `mask` arg removed)
# --------------------------------------------------------------------------- #

def _conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


def _upconv(in_channels, out_channels):
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)


class LUCIDNet(nn.Module):
    def __init__(self, input_channels=3, output_channels=1, text_dim=128, spatial_dim=512):
        super().__init__()
        self.enc1 = _conv_block(input_channels, 64)
        self.enc2 = _conv_block(64, 128)
        self.enc3 = _conv_block(128, 256)
        self.enc4 = _conv_block(256, 512)
        self.bottleneck = _conv_block(512, 1024)

        self.up4 = _upconv(1024, 512)
        self.dec4 = _conv_block(1024, 512)
        self.attn4 = CrossAttention(512, text_dim + spatial_dim)

        self.up3 = _upconv(512, 256)
        self.dec3 = _conv_block(512, 256)
        self.attn3 = CrossAttention(256, text_dim + spatial_dim)

        self.up2 = _upconv(256, 128)
        self.dec2 = _conv_block(256, 128)
        self.attn2 = CrossAttention(128, text_dim + spatial_dim)

        self.up1 = _upconv(128, 64)
        self.dec1 = _conv_block(128, 64)
        self.attn1 = CrossAttention(64, text_dim + spatial_dim)

        self.final = nn.Conv2d(64, 3, kernel_size=3, padding=1)
        self.final2 = nn.Conv2d(3, output_channels, kernel_size=1)
        self.out_act = nn.Sigmoid()

    def forward(self, x, text_embed, spatial_embed):
        cond_embed = torch.cat([text_embed, spatial_embed], dim=1)

        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, 2))
        enc3 = self.enc3(F.max_pool2d(enc2, 2))
        enc4 = self.enc4(F.max_pool2d(enc3, 2))
        bottleneck = self.bottleneck(F.max_pool2d(enc4, 2))

        up4 = self.up4(bottleneck)
        dec4 = self.dec4(torch.cat((up4, enc4), dim=1))
        dec4 = self.attn4(dec4, cond_embed)

        up3 = self.up3(dec4)
        dec3 = self.dec3(torch.cat((up3, enc3), dim=1))
        dec3 = self.attn3(dec3, cond_embed)

        up2 = self.up2(dec3)
        dec2 = self.dec2(torch.cat((up2, enc2), dim=1))
        dec2 = self.attn2(dec2, cond_embed)

        up1 = self.up1(dec2)
        dec1 = self.dec1(torch.cat((up1, enc1), dim=1))
        dec1 = self.attn1(dec1, cond_embed)

        out = self.final(dec1)
        return self.out_act(self.final2(x - out))


# --------------------------------------------------------------------------- #
# Model 2: Attention U-Net baseline (no text/spatial conditioning at all) —
# an ablation-style comparison for "does the multimodal conditioning even help".
# --------------------------------------------------------------------------- #

class AttentionGate(nn.Module):
    """Standard additive attention gate (Oktay et al., 2018)."""

    def __init__(self, gate_channels, skip_channels, inter_channels):
        super().__init__()
        self.w_gate = nn.Sequential(nn.Conv2d(gate_channels, inter_channels, 1), nn.BatchNorm2d(inter_channels))
        self.w_skip = nn.Sequential(nn.Conv2d(skip_channels, inter_channels, 1), nn.BatchNorm2d(inter_channels))
        self.psi = nn.Sequential(nn.Conv2d(inter_channels, 1, 1), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate, skip):
        g = self.w_gate(gate)
        s = self.w_skip(skip)
        if g.shape[-2:] != s.shape[-2:]:
            g = F.interpolate(g, size=s.shape[-2:], mode="bilinear", align_corners=False)
        alpha = self.psi(self.relu(g + s))
        return skip * alpha


class AttentionUNet(nn.Module):
    def __init__(self, input_channels=3, output_channels=1):
        super().__init__()
        self.enc1 = _conv_block(input_channels, 64)
        self.enc2 = _conv_block(64, 128)
        self.enc3 = _conv_block(128, 256)
        self.enc4 = _conv_block(256, 512)
        self.bottleneck = _conv_block(512, 1024)

        self.up4, self.gate4, self.dec4 = _upconv(1024, 512), AttentionGate(1024, 512, 256), _conv_block(1024, 512)
        self.up3, self.gate3, self.dec3 = _upconv(512, 256), AttentionGate(512, 256, 128), _conv_block(512, 256)
        self.up2, self.gate2, self.dec2 = _upconv(256, 128), AttentionGate(256, 128, 64), _conv_block(256, 128)
        self.up1, self.gate1, self.dec1 = _upconv(128, 64), AttentionGate(128, 64, 32), _conv_block(128, 64)

        self.final = nn.Conv2d(64, output_channels, kernel_size=1)
        self.out_act = nn.Sigmoid()

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, 2))
        enc3 = self.enc3(F.max_pool2d(enc2, 2))
        enc4 = self.enc4(F.max_pool2d(enc3, 2))
        bottleneck = self.bottleneck(F.max_pool2d(enc4, 2))

        up4 = self.up4(bottleneck)
        dec4 = self.dec4(torch.cat((up4, self.gate4(bottleneck, enc4)), dim=1))
        up3 = self.up3(dec4)
        dec3 = self.dec3(torch.cat((up3, self.gate3(dec4, enc3)), dim=1))
        up2 = self.up2(dec3)
        dec2 = self.dec2(torch.cat((up2, self.gate2(dec3, enc2)), dim=1))
        up1 = self.up1(dec2)
        dec1 = self.dec1(torch.cat((up1, self.gate1(dec2, enc1)), dim=1))

        return self.out_act(self.final(dec1))


# --------------------------------------------------------------------------- #
# Model 3: U-Net decoder over a timm pretrained (ImageNet) encoder —
# transfer learning, the strongest lever for tiny (35-80 image) datasets.
# --------------------------------------------------------------------------- #

class _DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = _conv_block(out_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class TimmEncoderUNet(nn.Module):
    """U-Net with a frozen-at-first, then fine-tuned, ImageNet-pretrained encoder."""

    def __init__(self, output_channels=1, backbone="resnet34", pretrained=True):
        super().__init__()
        if timm is None:
            raise ImportError("timm is required for TimmEncoderUNet")
        self.encoder = timm.create_model(backbone, pretrained=pretrained, features_only=True, in_chans=3)
        ch = self.encoder.feature_info.channels()  # e.g. [64, 64, 128, 256, 512] for resnet34
        self.bottleneck = _conv_block(ch[-1], ch[-1])
        self.dec4 = _DecoderBlock(ch[-1], ch[-2], ch[-2])
        self.dec3 = _DecoderBlock(ch[-2], ch[-3], ch[-3])
        self.dec2 = _DecoderBlock(ch[-3], ch[-4], ch[-4])
        self.dec1 = _DecoderBlock(ch[-4], ch[-5], ch[-5])
        self.final = nn.Conv2d(ch[-5], output_channels, kernel_size=1)
        self.out_act = nn.Sigmoid()

    def forward(self, x):
        feats = self.encoder(x)  # list of 5 feature maps, shallow -> deep
        f1, f2, f3, f4, f5 = feats
        b = self.bottleneck(f5)
        d4 = self.dec4(b, f4)
        d3 = self.dec3(d4, f3)
        d2 = self.dec2(d3, f2)
        d1 = self.dec1(d2, f1)
        out = self.final(d1)
        out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return self.out_act(out)


def build_model(name, vocab_size=None):
    if name == "lucidnet":
        return {
            "unet": LUCIDNet(),
            "text_encoder": TextEncoder(vocab_size=vocab_size, embed_dim=128),
            "spatial_encoder": SpatialEncoder(in_channels=1, base_channels=64),
        }
    if name == "attn_unet":
        return {"unet": AttentionUNet()}
    if name == "timm_unet":
        return {"unet": TimmEncoderUNet(backbone="resnet34", pretrained=True)}
    raise ValueError(f"Unknown model name: {name}")
