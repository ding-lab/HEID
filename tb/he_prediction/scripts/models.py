#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F


def _group_count(channels: int, maximum: int = 8) -> int:

    for groups in range(min(maximum, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class ConvNormAct(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
    ) -> None:
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )


class SeparableResidual(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            3,
            stride=stride,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        self.depthwise_norm = nn.GroupNorm(
            _group_count(in_channels), in_channels
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.pointwise_norm = nn.GroupNorm(
            _group_count(out_channels), out_channels
        )
        self.activation = nn.SiLU(inplace=True)
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    1,
                    stride=stride,
                    bias=False,
                ),
                nn.GroupNorm(_group_count(out_channels), out_channels),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = self.activation(self.depthwise_norm(self.depthwise(x)))
        x = self.pointwise_norm(self.pointwise(x))
        return self.activation(x + residual)


class EncoderStage(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        blocks: int,
        downsample: bool,
    ) -> None:
        if blocks < 1:
            raise ValueError("every encoder stage needs at least one block")
        layers: list[nn.Module] = [
            SeparableResidual(
                in_channels,
                out_channels,
                stride=2 if downsample else 1,
            )
        ]
        layers.extend(
            SeparableResidual(out_channels, out_channels)
            for _ in range(blocks - 1)
        )
        super().__init__(*layers)


class RegionEncoder(nn.Module):

    def __init__(
        self,
        widths: Sequence[int],
        blocks: Sequence[int],
        multiscale_rgb: bool = True,
    ) -> None:
        super().__init__()
        if len(widths) != 4 or len(blocks) != 4:
            raise ValueError("widths and blocks must each have four entries")
        if any(width < 8 for width in widths):
            raise ValueError("encoder widths must be at least eight")
        stem_mid = max(12, widths[0] // 2)
        self.stem = nn.Sequential(
            ConvNormAct(3, stem_mid, 3, stride=2),
            ConvNormAct(stem_mid, widths[0], 3, stride=2),
        )
        self.stages = nn.ModuleList(
            [
                EncoderStage(widths[0], widths[0], blocks[0], False),
                EncoderStage(widths[0], widths[1], blocks[1], True),
                EncoderStage(widths[1], widths[2], blocks[2], True),
                EncoderStage(widths[2], widths[3], blocks[3], True),
            ]
        )
        self.multiscale_rgb = bool(multiscale_rgb)
        self.rgb_projections = nn.ModuleList(
            nn.Conv2d(3, width, 1, bias=False) for width in widths
        )

    def forward(self, rgb: torch.Tensor) -> list[torch.Tensor]:
        x = self.stem(rgb)
        features: list[torch.Tensor] = []
        for stage_index, stage in enumerate(self.stages):
            x = stage(x)
            if self.multiscale_rgb:
                reduced_rgb = F.interpolate(
                    rgb,
                    size=x.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                x = x + self.rgb_projections[stage_index](reduced_rgb)
            features.append(x)
        return features


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.reduce = ConvNormAct(in_channels, out_channels, kernel_size=1)
        self.fuse = nn.Sequential(
            SeparableResidual(out_channels + skip_channels, out_channels),
            SeparableResidual(out_channels, out_channels),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(
            self.reduce(x),
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return self.fuse(torch.cat((x, skip), dim=1))


def resample_to(x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:

    height, width = int(x.shape[-2]), int(x.shape[-1])
    target_height, target_width = int(size[0]), int(size[1])
    if (height, width) == (target_height, target_width):
        return x
    if height >= target_height and width >= target_width:
        if height % target_height == 0 and width % target_width == 0:
            stride = (height // target_height, width // target_width)
            return F.avg_pool2d(x, kernel_size=stride, stride=stride)
        return F.adaptive_avg_pool2d(x, (target_height, target_width))
    return F.interpolate(
        x, size=(target_height, target_width), mode="bilinear", align_corners=False
    )


class OutputHeads(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        hidden = max(16, in_channels)
        self.shared = SeparableResidual(in_channels, hidden)
        self.tumor = nn.Conv2d(hidden, 1, 1)
        self.sdf = nn.Conv2d(hidden, 1, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.shared(x)
        return {
            "tumor_logits": self.tumor(x),


            "sdf_normalized": torch.tanh(self.sdf(x)),
        }


class RegionUNet(nn.Module):

    output_stride = 4

    def __init__(
        self,
        widths: Sequence[int],
        blocks: Sequence[int],
        multiscale_rgb: bool = True,
        label_size: int | None = None,
    ) -> None:
        super().__init__()
        self.encoder = RegionEncoder(widths, blocks, multiscale_rgb)
        self.decode_2 = DecoderBlock(widths[3], widths[2], widths[2])
        self.decode_1 = DecoderBlock(widths[2], widths[1], widths[1])
        self.decode_0 = DecoderBlock(widths[1], widths[0], widths[0])
        self.heads = OutputHeads(widths[0])
        self.label_size = int(label_size) if label_size else None

    def forward(self, rgb: torch.Tensor) -> dict[str, torch.Tensor]:
        feature_0, feature_1, feature_2, feature_3 = self.encoder(rgb)
        x = self.decode_2(feature_3, feature_2)
        x = self.decode_1(x, feature_1)
        x = self.decode_0(x, feature_0)
        if self.label_size:
            x = resample_to(x, (self.label_size, self.label_size))
        return self.heads(x)


class RegionFPN(nn.Module):

    output_stride = 4

    def __init__(
        self,
        widths: Sequence[int],
        blocks: Sequence[int],
        fpn_width: int,
        multiscale_rgb: bool = True,
        label_size: int | None = None,
    ) -> None:
        super().__init__()
        self.encoder = RegionEncoder(widths, blocks, multiscale_rgb)
        self.lateral = nn.ModuleList(
            nn.Conv2d(width, fpn_width, 1, bias=False) for width in widths
        )
        self.refine = nn.Sequential(
            SeparableResidual(fpn_width, fpn_width),
            SeparableResidual(fpn_width, fpn_width),
        )
        self.heads = OutputHeads(fpn_width)
        self.label_size = int(label_size) if label_size else None

    def forward(self, rgb: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.encoder(rgb)
        target_size = features[0].shape[-2:]
        pyramid = self.lateral[0](features[0])
        for projection, feature in zip(self.lateral[1:], features[1:]):
            pyramid = pyramid + F.interpolate(
                projection(feature),
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
        if self.label_size:
            pyramid = resample_to(pyramid, (self.label_size, self.label_size))
        return self.heads(self.refine(pyramid))


class PretrainedFPN(nn.Module):

    output_stride = 4

    def __init__(
        self,
        encoder_name: str,
        fpn_width: int,
        label_size: int,
        pretrained: bool = True,
        drop_path_rate: float = 0.0,
        normalization: Sequence[float] | None = None,
        input_normalization: Sequence[float] | None = None,
    ) -> None:
        super().__init__()
        import timm

        self.encoder = timm.create_model(
            encoder_name,
            pretrained=bool(pretrained),
            features_only=True,
            out_indices=(0, 1, 2, 3),
            drop_path_rate=float(drop_path_rate),
        )
        channels = list(self.encoder.feature_info.channels())
        self.lateral = nn.ModuleList(
            nn.Conv2d(count, fpn_width, 1, bias=False) for count in channels
        )
        self.refine = nn.Sequential(
            SeparableResidual(fpn_width, fpn_width),
            SeparableResidual(fpn_width, fpn_width),
        )
        self.heads = OutputHeads(fpn_width)
        self.label_size = int(label_size)


        pipeline = tuple(
            float(value) for value in (input_normalization or (0.5, 0.5, 0.5, 0.5, 0.5, 0.5))
        )
        encoder_statistics = tuple(
            float(value)
            for value in (
                normalization
                or (0.485, 0.456, 0.406, 0.229, 0.224, 0.225)
            )
        )
        self.register_buffer(
            "pipeline_mean",
            torch.tensor(pipeline[:3]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "pipeline_std",
            torch.tensor(pipeline[3:]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "encoder_mean",
            torch.tensor(encoder_statistics[:3]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "encoder_std",
            torch.tensor(encoder_statistics[3:]).view(1, 3, 1, 1),
            persistent=False,
        )

    def renormalize(self, rgb: torch.Tensor) -> torch.Tensor:
        unit = rgb * self.pipeline_std.to(rgb.dtype) + self.pipeline_mean.to(
            rgb.dtype
        )
        return (unit - self.encoder_mean.to(rgb.dtype)) / self.encoder_std.to(
            rgb.dtype
        )

    def forward(self, rgb: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.encoder(self.renormalize(rgb))
        size = (self.label_size, self.label_size)
        pyramid = resample_to(self.lateral[0](features[0]), size)
        for projection, feature in zip(self.lateral[1:], features[1:]):
            pyramid = pyramid + resample_to(projection(feature), size)
        return self.heads(self.refine(pyramid))


@dataclass(frozen=True)
class ModelSummary:
    architecture: str
    trainable_parameters: int
    output_stride: int


def build_model(candidate: Mapping[str, Any]) -> nn.Module:

    architecture = str(candidate["architecture"])
    label_size = candidate.get("label_size")
    if architecture == "pretrained_fpn":
        if label_size is None:
            raise ValueError("pretrained_fpn requires an explicit label_size")
        return PretrainedFPN(
            encoder_name=str(candidate["encoder"]),
            fpn_width=int(candidate["fpn_width"]),
            label_size=int(label_size),
            pretrained=bool(candidate.get("pretrained", True)),
            drop_path_rate=float(candidate.get("drop_path_rate", 0.0)),
            normalization=candidate.get("encoder_normalization"),
            input_normalization=candidate.get("pipeline_normalization"),
        )
    widths = tuple(int(value) for value in candidate["widths"])
    blocks = tuple(int(value) for value in candidate["blocks"])
    multiscale_rgb = bool(candidate.get("multiscale_rgb", True))
    if architecture == "region_unet":
        model: nn.Module = RegionUNet(
            widths, blocks, multiscale_rgb, label_size=label_size
        )
    elif architecture == "region_fpn":
        model = RegionFPN(
            widths,
            blocks,
            fpn_width=int(candidate["fpn_width"]),
            multiscale_rgb=multiscale_rgb,
            label_size=label_size,
        )
    else:
        raise ValueError(f"unsupported architecture: {architecture}")
    return model


def summarize_model(model: nn.Module, architecture: str) -> ModelSummary:
    return ModelSummary(
        architecture=architecture,
        trainable_parameters=sum(
            parameter.numel() for parameter in model.parameters()
            if parameter.requires_grad
        ),
        output_stride=int(getattr(model, "output_stride")),
    )


__all__ = [
    "ModelSummary",
    "PretrainedFPN",
    "RegionFPN",
    "RegionUNet",
    "build_model",
    "resample_to",
    "summarize_model",
]
