"""RecurrentCNN / PlainCNN baselines (weight-shared looped conv processors).

RecurrentCNN and PlainCNN are copied verbatim from
CF_Milestone2_Week1_HF.ipynb Section 5 (identical to maze_cf.RecurrentCNN
from Milestone 1); logits_at is the package addition used to probe several
test-time horizons from a single rollout.
"""
import torch
import torch.nn as nn


class RecurrentCNN(nn.Module):
    # weight-shared residual conv cell with per-step input injection
    # (identical to maze_cf.RecurrentCNN from Milestone 1)
    def __init__(self, in_ch=3, hid=22):
        super().__init__()
        self.encoder = nn.Conv2d(in_ch, hid, 3, padding=1)
        self.cell = nn.Sequential(
            nn.Conv2d(in_ch + hid, hid, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hid, hid, 3, padding=1),
        )
        self.readout = nn.Conv2d(hid, 1, 1)

    def forward(self, x, T=8):
        h = self.encoder(x)
        for _ in range(T):
            h = h + self.cell(torch.cat([x, h], dim=1))
        return self.readout(h).squeeze(1)

    def logits_at(self, x, Ts):
        # ONE rollout to max(Ts); returns {T: logits (B, H, W)}, read out after
        # every requested step count. logits_at(x, Ts)[T] == forward(x, T).
        Ts = sorted(set(Ts))
        h = self.encoder(x)
        out = {}
        t = 0
        for T in Ts:
            for _ in range(T - t):
                h = h + self.cell(torch.cat([x, h], dim=1))
            t = T
            out[T] = self.readout(h).squeeze(1)
        return out


class PlainCNN(nn.Module):
    # 5 x (3x3 conv) + 1x1 readout -> receptive field 11x11 (+/-5 cells); no loop, no state
    def __init__(self, in_ch=3, hid=18, depth=6):
        super().__init__()
        layers = [nn.Conv2d(in_ch, hid, 3, padding=1), nn.GELU()]
        for _ in range(depth - 2):
            layers += [nn.Conv2d(hid, hid, 3, padding=1), nn.GELU()]
        layers += [nn.Conv2d(hid, 1, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x, T=1):            # T accepted and ignored
        return self.net(x).squeeze(1)


def n_params(model):
    return sum(p.numel() for p in model.parameters())
