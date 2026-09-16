
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn

from models import OutputHeads, SeparableResidual, resample_to


DEFAULT_LAYERS = (6, 12, 18, 24)


class PathologyViTFPN(nn.Module):

    output_stride = 4

    def __init__(
        self,
        encoder_name: str,
        fpn_width: int,
        label_size: int,
        pretrained: bool = True,
        layers: Sequence[int] = DEFAULT_LAYERS,
        gradient_checkpointing: bool = True,
        input_size: int = 1024,
        patch_size: int | None = None,
        input_upsample: int = 1,
        normalization: Sequence[float] | None = None,
        input_normalization: Sequence[float] | None = None,
    ) -> None:
        super().__init__()
        from transformers import AutoConfig, AutoModel


        attention = {"attn_implementation": "sdpa"}
        if pretrained:
            self.encoder = AutoModel.from_pretrained(encoder_name, **attention)
        else:


            self.encoder = AutoModel.from_config(
                AutoConfig.from_pretrained(encoder_name), **attention
            )
        if gradient_checkpointing:
            self.encoder.gradient_checkpointing_enable()

        hidden = int(self.encoder.config.hidden_size)
        self.patch_size = int(self.encoder.config.patch_size)
        self.input_upsample = int(input_upsample)
        if patch_size is not None:
            self._resize_patch_embedding(int(patch_size))
        self._resize_position_embeddings(int(input_size) * self.input_upsample)
        self.layers = tuple(int(index) for index in layers)
        depth = int(self.encoder.config.num_hidden_layers)
        if not all(1 <= index <= depth for index in self.layers):
            raise ValueError(f"layer indices {self.layers} outside encoder depth {depth}")

        self.lateral = nn.ModuleList(nn.Conv2d(hidden, fpn_width, 1, bias=False) for _ in self.layers)
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
            for value in (normalization or (0.485, 0.456, 0.406, 0.229, 0.224, 0.225))
        )
        for name, values in (
            ("pipeline_mean", pipeline[:3]),
            ("pipeline_std", pipeline[3:]),
            ("encoder_mean", encoder_statistics[:3]),
            ("encoder_std", encoder_statistics[3:]),
        ):
            self.register_buffer(name, torch.tensor(values).view(1, 3, 1, 1), persistent=False)

    def _resize_patch_embedding(self, target: int) -> None:

        embeddings = self.encoder.embeddings.patch_embeddings
        convolution = embeddings.projection
        current = int(convolution.kernel_size[0])
        if current == target:
            return
        with torch.no_grad():
            weight = nn.functional.interpolate(
                convolution.weight.float(), size=(target, target), mode="bicubic", align_corners=False
            ) * (current**2 / target**2)
        replacement = nn.Conv2d(
            convolution.in_channels,
            convolution.out_channels,
            kernel_size=target,
            stride=target,
            bias=convolution.bias is not None,
        )
        replacement.weight = nn.Parameter(weight.to(convolution.weight.dtype))
        if convolution.bias is not None:
            replacement.bias = nn.Parameter(convolution.bias.detach().clone())
        embeddings.projection = replacement
        embeddings.patch_size = target
        self.encoder.config.patch_size = target
        self.patch_size = target

    def _resize_position_embeddings(self, input_size: int) -> None:

        embeddings = self.encoder.embeddings
        table = embeddings.position_embeddings
        side = input_size // self.patch_size
        wanted = side * side + 1
        if table.shape[1] == wanted:
            return
        with torch.no_grad():
            class_token, patches = table[:, :1], table[:, 1:]
            grid = int(round(patches.shape[1] ** 0.5))
            if grid * grid != patches.shape[1]:
                raise ValueError(f"position table of {patches.shape[1]} is not square")
            resized = torch.nn.functional.interpolate(
                patches.reshape(1, grid, grid, -1).permute(0, 3, 1, 2).float(),
                size=(side, side),
                mode="bicubic",
                align_corners=False,
            )
            resized = resized.permute(0, 2, 3, 1).reshape(1, side * side, -1)
            baked = torch.cat([class_token.float(), resized], dim=1).to(table.dtype)
        embeddings.position_embeddings = nn.Parameter(baked, requires_grad=table.requires_grad)
        self.position_tokens = wanted

    def renormalize(self, rgb: torch.Tensor) -> torch.Tensor:

        unit = rgb * self.pipeline_std.to(rgb.dtype) + self.pipeline_mean.to(rgb.dtype)
        return (unit - self.encoder_mean.to(rgb.dtype)) / self.encoder_std.to(rgb.dtype)

    def forward(self, rgb: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.input_upsample != 1:


            rgb = nn.functional.interpolate(
                rgb, scale_factor=self.input_upsample, mode="bilinear", align_corners=False
            )
        height, width = rgb.shape[-2:]
        if height % self.patch_size or width % self.patch_size:
            raise ValueError(
                f"input {height}x{width} is not a multiple of patch {self.patch_size}; "
                "token grid would not divide the label grid exactly"
            )
        rows, columns = height // self.patch_size, width // self.patch_size

        states = self.encoder(
            pixel_values=self.renormalize(rgb), output_hidden_states=True
        ).hidden_states

        size = (self.label_size, self.label_size)
        pyramid: torch.Tensor | None = None
        for projection, index in zip(self.lateral, self.layers):
            tokens = states[index][:, 1:, :]
            grid = tokens.transpose(1, 2).reshape(tokens.shape[0], -1, rows, columns)
            level = resample_to(projection(grid), size)
            pyramid = level if pyramid is None else pyramid + level
        assert pyramid is not None
        return self.heads(self.refine(pyramid))


def build_fm_model(candidate: Mapping[str, Any]) -> nn.Module:

    architecture = str(candidate["architecture"])
    if architecture != "pathology_vit_fpn":
        raise ValueError(f"unsupported foundation-model architecture: {architecture}")
    label_size = candidate.get("label_size")
    if label_size is None:
        raise ValueError("pathology_vit_fpn requires an explicit label_size")
    return PathologyViTFPN(
        encoder_name=str(candidate["encoder"]),
        fpn_width=int(candidate["fpn_width"]),
        label_size=int(label_size),
        pretrained=bool(candidate.get("pretrained", True)),
        layers=candidate.get("encoder_layers", DEFAULT_LAYERS),
        input_size=int(candidate.get("input_size", 1024)),
        patch_size=candidate.get("patch_size"),
        input_upsample=int(candidate.get("input_upsample", 1)),
        gradient_checkpointing=bool(candidate.get("gradient_checkpointing", True)),
        normalization=candidate.get("encoder_normalization"),
        input_normalization=candidate.get("pipeline_normalization"),
    )


def is_foundation_candidate(candidate: Mapping[str, Any]) -> bool:
    return str(candidate.get("architecture", "")) == "pathology_vit_fpn"


__all__ = ["PathologyViTFPN", "build_fm_model", "is_foundation_candidate"]
