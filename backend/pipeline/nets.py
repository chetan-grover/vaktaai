"""Torch modules shared by inference (pipeline/) and training (training/).

Imported lazily: nothing here runs at pipeline import time and torch stays optional.
- PhysNet: rPPG-Toolbox's PhysNet_padding_Encoder_Decoder_MAX (E1), for its pretrained weights.
- GazeSmoother: GRU over per-frame gaze angles -> 3x3 audience cell (E2).
- CrossAttentionFusionHead: audio <-> face cross-attention over V-A embeddings (E3).
- L2CS: ResNet-50 gaze regressor with binned yaw/pitch heads (E2, Gaze360 weights).
"""
from __future__ import annotations

import torch
from torch import nn


# ----------------------------------------------------------------------------- E1
def _conv3d(cin: int, cout: int, kernel, padding) -> nn.Sequential:
    return nn.Sequential(nn.Conv3d(cin, cout, kernel, stride=1, padding=padding),
                         nn.BatchNorm3d(cout), nn.ReLU(inplace=True))


class PhysNet(nn.Module):
    """Input (B, 3, T, 72, 72) in [0,1] -> rPPG waveform (B, T). Layer names match rPPG-Toolbox."""

    def __init__(self, frames: int = 128) -> None:
        super().__init__()
        self.ConvBlock1 = _conv3d(3, 16, [1, 5, 5], [0, 2, 2])
        self.ConvBlock2 = _conv3d(16, 32, [3, 3, 3], 1)
        self.ConvBlock3 = _conv3d(32, 64, [3, 3, 3], 1)
        self.ConvBlock4 = _conv3d(64, 64, [3, 3, 3], 1)
        self.ConvBlock5 = _conv3d(64, 64, [3, 3, 3], 1)
        self.ConvBlock6 = _conv3d(64, 64, [3, 3, 3], 1)
        self.ConvBlock7 = _conv3d(64, 64, [3, 3, 3], 1)
        self.ConvBlock8 = _conv3d(64, 64, [3, 3, 3], 1)
        self.ConvBlock9 = _conv3d(64, 64, [3, 3, 3], 1)
        self.upsample = nn.Sequential(
            nn.ConvTranspose3d(64, 64, [4, 1, 1], stride=[2, 1, 1], padding=[1, 0, 0]),
            nn.BatchNorm3d(64), nn.ELU())
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose3d(64, 64, [4, 1, 1], stride=[2, 1, 1], padding=[1, 0, 0]),
            nn.BatchNorm3d(64), nn.ELU())
        self.ConvBlock10 = nn.Conv3d(64, 1, [1, 1, 1], stride=1, padding=0)
        self.MaxpoolSpa = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))
        self.MaxpoolSpaTem = nn.MaxPool3d((2, 2, 2), stride=2)
        self.poolspa = nn.AdaptiveAvgPool3d((frames, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.shape[2]
        x = self.MaxpoolSpa(self.ConvBlock1(x))
        x = self.MaxpoolSpaTem(self.ConvBlock3(self.ConvBlock2(x)))
        x = self.MaxpoolSpaTem(self.ConvBlock5(self.ConvBlock4(x)))
        x = self.MaxpoolSpa(self.ConvBlock7(self.ConvBlock6(x)))
        x = self.ConvBlock9(self.ConvBlock8(x))
        x = self.upsample2(self.upsample(x))
        x = self.ConvBlock10(self.poolspa(x))
        return x.view(-1, length)


# ----------------------------------------------------------------------------- E2
class GazeSmoother(nn.Module):
    """Bidirectional GRU: (B, T, F) gaze features -> (B, T, 9) audience-cell logits."""

    def __init__(self, in_features: int = 4, hidden: int = 32, cells: int = 9) -> None:
        super().__init__()
        self.gru = nn.GRU(in_features, hidden, num_layers=1, batch_first=True, bidirectional=True)
        self.head = nn.Linear(2 * hidden, cells)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.head(out)


class L2CS(nn.Module):
    """L2CS-Net (ResNet-50 backbone, 90 bins of 4 degrees per angle)."""

    def __init__(self, bins: int = 90) -> None:
        super().__init__()
        import torchvision

        resnet = torchvision.models.resnet50(weights=None)
        self.conv1, self.bn1, self.relu, self.maxpool = resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_yaw_gaze = nn.Linear(2048, bins)
        self.fc_pitch_gaze = nn.Linear(2048, bins)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = torch.flatten(self.avgpool(x), 1)
        return self.fc_yaw_gaze(x), self.fc_pitch_gaze(x)

    @staticmethod
    def to_degrees(logits: torch.Tensor) -> torch.Tensor:
        idx = torch.arange(logits.shape[1], dtype=torch.float32, device=logits.device)
        return (torch.softmax(logits, dim=1) * idx).sum(dim=1) * 4 - 180


# ----------------------------------------------------------------------------- E3
class _CrossBlock(nn.Module):
    """Each modality attends to the other, then a feed-forward layer (pre-norm)."""

    def __init__(self, d: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.norm_a, self.norm_f = nn.LayerNorm(d), nn.LayerNorm(d)
        self.a_from_f = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.f_from_a = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.ff_a = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * d, d))
        self.ff_f = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * d, d))

    def forward(self, a, f, a_mask=None, f_mask=None):
        na, nf = self.norm_a(a), self.norm_f(f)
        a = a + self.a_from_f(na, nf, nf, key_padding_mask=f_mask, need_weights=False)[0]
        f = f + self.f_from_a(nf, na, na, key_padding_mask=a_mask, need_weights=False)[0]
        return a + self.ff_a(a), f + self.ff_f(f)


def _masked_mean(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return x.mean(dim=1)
    keep = (~mask).unsqueeze(-1).float()
    return (x * keep).sum(dim=1) / keep.sum(dim=1).clamp(min=1.0)


class CrossAttentionFusionHead(nn.Module):
    """Audio (B, Ta, d_audio) + face (B, Tf, d_face) sequences ->
    emotion logits, fused V-A, per-modality V-A and a mismatch logit.

    Congruence distance = || audio_va - face_va ||; the mismatch head is trained on
    shuffled audio/face pairs from different emotions (see training/e3_fusion.ipynb).
    """

    def __init__(self, d_audio: int, d_face: int, d_model: int = 128, heads: int = 4,
                 layers: int = 2, n_emotions: int = 8, dropout: float = 0.1, max_len: int = 64) -> None:
        super().__init__()
        self.config = dict(d_audio=d_audio, d_face=d_face, d_model=d_model, heads=heads,
                           layers=layers, n_emotions=n_emotions, dropout=dropout, max_len=max_len)
        self.proj_a = nn.Sequential(nn.LayerNorm(d_audio), nn.Linear(d_audio, d_model))
        self.proj_f = nn.Sequential(nn.LayerNorm(d_face), nn.Linear(d_face, d_model))
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        self.blocks = nn.ModuleList([_CrossBlock(d_model, heads, dropout) for _ in range(layers)])
        self.va_audio = nn.Linear(d_model, 2)
        self.va_face = nn.Linear(d_model, 2)
        self.emotion = nn.Linear(2 * d_model, n_emotions)
        self.va_fused = nn.Linear(2 * d_model, 2)
        self.mismatch = nn.Linear(2 * d_model + 2, 1)

    def forward(self, audio, face, audio_mask=None, face_mask=None) -> dict[str, torch.Tensor]:
        ta, tf = audio.shape[1], face.shape[1]
        a = self.proj_a(audio) + self.pos[:, :ta]
        f = self.proj_f(face) + self.pos[:, :tf]
        a_uni, f_uni = _masked_mean(a, audio_mask), _masked_mean(f, face_mask)
        for block in self.blocks:
            a, f = block(a, f, audio_mask, face_mask)
        pooled = torch.cat([_masked_mean(a, audio_mask), _masked_mean(f, face_mask)], dim=-1)
        va_a, va_f = torch.tanh(self.va_audio(a_uni)), torch.tanh(self.va_face(f_uni))
        gap = (va_a - va_f).abs()
        return {
            "emotion_logits": self.emotion(pooled),
            "va": torch.tanh(self.va_fused(pooled)),
            "audio_va": va_a,
            "face_va": va_f,
            "mismatch_logit": self.mismatch(torch.cat([pooled, gap], dim=-1)).squeeze(-1),
        }
