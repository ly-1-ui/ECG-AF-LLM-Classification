from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """
    Two Conv1d layers with ReLU.
    Default kernel_size=3 matches the baseline stream-1 setting.
    """

    def __init__(self, ch_in: int, ch_out: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv1d(ch_in, ch_out, kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv1d(ch_out, ch_out, kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class TripleConv(nn.Module):
    """
    Three Conv1d layers with ReLU, kernel_size=3.
    Matches the baseline setting used in later layers.
    """

    def __init__(self, ch_in: int, ch_out: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(ch_in, ch_out, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(ch_out, ch_out, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(ch_out, ch_out, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class MLP(nn.Module):
    """
    MLP head used by Task1 classifier:
    feature_dim -> 1024 -> 1024 -> 256 -> ch_out
    """

    def __init__(self, ch_in: int, ch_out: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(ch_in, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, ch_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class Mscnn(nn.Module):
    """
    Multi-stream 1D CNN.

    Input:
        x: [B, C_in, T] (typically T=2400)

    Outputs:
        - forward_features(x): [B, feature_dim] (for Task2 multimodal LLM)
        - forward(x): [B, ch_out] with sigmoid (for Task1 classification)
    """

    def __init__(
        self,
        ch_in: int,
        ch_out: int,
        use_stream2: bool = True,
        stream2_kernel: int = 7,
        input_len: Optional[int] = 2400,
    ):
        super().__init__()
        self.use_stream2 = use_stream2
        self.ch_in = ch_in

        # Stream 1 (kernel=3)
        self.conv11 = DoubleConv(ch_in, 64, kernel_size=3)
        self.pool11 = nn.MaxPool1d(3, stride=3)
        self.conv12 = DoubleConv(64, 128, kernel_size=3)
        self.pool12 = nn.MaxPool1d(3, stride=3)
        self.conv13 = TripleConv(128, 256)
        self.pool13 = nn.MaxPool1d(2, stride=2)
        self.conv14 = TripleConv(256, 512)
        self.pool14 = nn.MaxPool1d(2, stride=2)
        self.conv15 = TripleConv(512, 512)
        self.pool15 = nn.MaxPool1d(2, stride=2)

        # Stream 2 (kernel=stream2_kernel for first two blocks, then kernel=3)
        if self.use_stream2:
            k = int(stream2_kernel)
            self.conv21 = DoubleConv(ch_in, 64, kernel_size=k)
            self.pool21 = nn.MaxPool1d(3, stride=3)
            self.conv22 = DoubleConv(64, 128, kernel_size=k)
            self.pool22 = nn.MaxPool1d(3, stride=3)
            self.conv23 = TripleConv(128, 256)
            self.pool23 = nn.MaxPool1d(2, stride=2)
            self.conv24 = TripleConv(256, 512)
            self.pool24 = nn.MaxPool1d(2, stride=2)
            self.conv25 = TripleConv(512, 512)
            self.pool25 = nn.MaxPool1d(2, stride=2)

        # Feature dimension for T=2400:
        # 2400 -> /3 -> 800 -> /3 -> 266 -> /2 -> 133 -> /2 -> 66 -> /2 -> 33
        # Single stream: 512 * 33 = 16896
        # Two streams: 16896 * 2 = 33792
        if input_len is None:
            self.feature_dim = 33792 if self.use_stream2 else 16896
        else:
            self.feature_dim = self._infer_feature_dim_runtime(input_len)

        # Task1 head
        self.out = MLP(self.feature_dim, ch_out)

    def _infer_feature_dim_runtime(self, seq_len: int) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, self.ch_in, seq_len, dtype=torch.float32)
            feat = self.forward_features(dummy)
        return int(feat.shape[1])

    @torch.no_grad()
    def infer_feature_dim(self, seq_len: Optional[int] = None) -> int:
        if seq_len is not None:
            return self._infer_feature_dim_runtime(seq_len)
        return int(self.feature_dim)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return the flattened CNN feature vector before the MLP head.
        """
        # Stream 1
        c11 = self.conv11(x)
        p11 = self.pool11(c11)
        c12 = self.conv12(p11)
        p12 = self.pool12(c12)
        c13 = self.conv13(p12)
        p13 = self.pool13(c13)
        c14 = self.conv14(p13)
        p14 = self.pool14(c14)
        c15 = self.conv15(p14)
        p15 = self.pool15(c15)

        merge = p15.view(p15.size(0), -1)

        # Stream 2
        if self.use_stream2:
            c21 = self.conv21(x)
            p21 = self.pool21(c21)
            c22 = self.conv22(p21)
            p22 = self.pool22(c22)
            c23 = self.conv23(p22)
            p23 = self.pool23(c23)
            c24 = self.conv24(p23)
            p24 = self.pool24(c24)
            c25 = self.conv25(p24)
            p25 = self.pool25(c25)

            merge2 = p25.view(p25.size(0), -1)
            merge = torch.cat((merge, merge2), dim=1)

        return merge

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Task1 forward: features -> MLP -> sigmoid.
        """
        feat = self.forward_features(x)
        logits = self.out(feat)
        probs = torch.sigmoid(logits)
        return probs


def freeze_module(module: nn.Module) -> None:
    """
    Freeze all parameters in the module (no gradients).
    """
    for p in module.parameters():
        p.requires_grad = False


def _extract_state_dict(ckpt: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """
    Handle common checkpoint formats.
    """
    if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        return ckpt["state_dict"]
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        return ckpt["model"]
    if "model_state_dict" in ckpt and isinstance(ckpt["model_state_dict"], dict):
        return ckpt["model_state_dict"]
    return ckpt  # assume it's already a state_dict


def load_mscnn_checkpoint(
    model: nn.Module,
    ckpt_path: Union[str, Path],
    map_location: Union[str, torch.device] = "cpu",
    strict: bool = True,
    remove_prefix: Optional[str] = None,
) -> None:
    """
    Load weights into Mscnn.

    Args:
        model: Mscnn instance.
        ckpt_path: path to model.pth or a checkpoint dict.
        map_location: torch.load map_location.
        strict: strict loading.
        remove_prefix: if state_dict keys have a prefix (e.g., 'module.'), remove it.
    """
    ckpt_path = Path(ckpt_path)
    obj = torch.load(str(ckpt_path), map_location=map_location)

    if isinstance(obj, dict):
        state = _extract_state_dict(obj)
    else:
        raise ValueError(f"Unsupported checkpoint type: {type(obj)}")

    if remove_prefix:
        new_state: Dict[str, torch.Tensor] = {}
        for k, v in state.items():
            if k.startswith(remove_prefix):
                new_state[k[len(remove_prefix) :]] = v
            else:
                new_state[k] = v
        state = new_state
    else:
        # Common DataParallel prefix
        if any(k.startswith("module.") for k in state.keys()):
            state = {k.replace("module.", "", 1): v for k, v in state.items()}

    model.load_state_dict(state, strict=strict)
