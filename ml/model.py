"""Spectrum -> RF program network.

Input:  the measured total spectrum on the 500-bin grid, plus its mirror (R -> -R),
        because every frequency is coupled to its mirror (the two overlapping
        transitions belong to mirror-orientation spins); optionally a third,
        constant channel with the normalized time since the burn.
Output: (a) logits over bins -> softmax gives the frequency profile of RF power;
        (b) total power (U, positive) and (c) log duration.
The program is U_j = total * softmax(logits)_j applied for duration T.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(cin, cout, 5, padding=2), nn.GroupNorm(8, cout), nn.GELU(),
                                 nn.Conv1d(cout, cout, 5, padding=2), nn.GroupNorm(8, cout), nn.GELU())

    def forward(self, x):
        return self.net(x)


class SpectrumToPulse(nn.Module):
    def __init__(self, n_bins=500, width=32, dropout=0.1, in_channels=2, prior_profile=None):
        """prior_profile: optional mean target profile. The network then learns a correction
        to it (logits = log prior + learned term, starting at zero), so before training it
        already reproduces the mean program."""
        super().__init__()
        w = width
        self.enc1 = ConvBlock(in_channels, w)
        self.enc2 = ConvBlock(w, 2*w)
        self.enc3 = ConvBlock(2*w, 4*w)
        self.mid = ConvBlock(4*w, 4*w)
        self.dec3 = ConvBlock(8*w, 2*w)
        self.dec2 = ConvBlock(4*w, w)
        self.dec1 = ConvBlock(2*w, w)
        self.profile = nn.Conv1d(w, 1, 1)
        self.drop = nn.Dropout(dropout)
        self.scalars = nn.Sequential(nn.Linear(4*w*2, 64), nn.GELU(), nn.Linear(64, 2))
        self.n_bins = n_bins
        if prior_profile is None:
            self.prior_logits = None
        else:
            prior = torch.as_tensor(prior_profile, dtype=torch.float32)
            self.register_buffer('prior_logits', torch.log(prior.clamp_min(1e-8)/prior.sum()))
            nn.init.zeros_(self.profile.weight); nn.init.zeros_(self.profile.bias)

    def forward(self, x):
        e1 = self.enc1(x)                       # N
        e2 = self.enc2(F.max_pool1d(e1, 2))     # N/2
        e3 = self.enc3(F.max_pool1d(e2, 2))     # N/4
        m = self.mid(F.max_pool1d(e3, 2))       # N/8
        g = torch.cat([m.mean(-1), m.amax(-1)], dim=1)
        s = self.scalars(self.drop(g))
        u3 = F.interpolate(m, size=e3.shape[-1], mode='linear', align_corners=False)
        d3 = self.dec3(torch.cat([u3, e3], 1))
        u2 = F.interpolate(d3, size=e2.shape[-1], mode='linear', align_corners=False)
        d2 = self.dec2(torch.cat([u2, e2], 1))
        u1 = F.interpolate(d2, size=e1.shape[-1], mode='linear', align_corners=False)
        d1 = self.dec1(torch.cat([u1, e1], 1))
        logits = self.profile(self.drop(d1)).squeeze(1)
        if self.prior_logits is not None:
            logits = logits+self.prior_logits
        return logits, s[:, 0], s[:, 1]         # profile logits, total-power (standardized), log T (standardized)
