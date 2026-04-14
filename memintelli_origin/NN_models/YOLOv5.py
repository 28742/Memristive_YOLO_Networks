# -*- coding:utf-8 -*-
# @File  : YOLOv5.py
# @Author: Zhou (extended by MemIntelli contributors)
# @Date  : 2026/01/15
"""
YOLOv5 wrapper for MemIntelli.

Notes
-----
- YOLOv5 official pretrained weights are typically distributed as `.pt` files that depend on
  the Ultralytics YOLOv5 codebase for loading.
- To avoid vendoring the full YOLOv5 implementation into MemIntelli, we load models via
  `torch.hub.load('ultralytics/yolov5', ...)`.

This module provides a `YOLOv5` nn.Module wrapper and a `YOLOv5_zoo` factory, mirroring the
style of other models in `memintelli/NN_models/`.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, List, Union, Set

import torch
import torch.nn as nn

from memintelli.NN_layers import Conv2dMem, LinearMem

# Pretrained model identifiers (resolved by torch.hub)
model_names = ("yolov5n", "yolov5s", "yolov5m")

def _to_int2(x):
    # Normalize scalar/tuple to 2-tuple of ints (for stride/padding/dilation)
    if isinstance(x, (tuple, list)):
        return int(x[0]), int(x[1])
    return int(x), int(x)


def _replace_conv2d_with_mem(
    module: nn.Module,
    *,
    engine: Any,
    input_slice: Any,
    weight_slice: Any,
    device: Any,
    bw_e: Any,
    input_paral_size: Any,
    weight_paral_size: Any,
    input_quant_gran: Any,
    weight_quant_gran: Any,
    target_idx: Optional[Union[int, List[int], Dict[int, Any]]] = None,
    counter_ctx: Optional[Dict[str, int]] = None,
) -> Dict[str, int]:
    """
    Recursively replace nn.Conv2d (groups==1, square kernels) with Conv2dMem.
    Returns counters for converted/skipped convs.
    """
    if counter_ctx is None:
        counter_ctx = {"count": 0}

    converted = 0
    skipped = 0

    for name, child in list(module.named_children()):
        if isinstance(child, nn.Conv2d):
            k_h, k_w = _to_int2(child.kernel_size)
            s_h, s_w = _to_int2(child.stride)
            p_h, p_w = _to_int2(child.padding)
            d_h, d_w = _to_int2(child.dilation)

            # Conv2dMem currently supports: groups==1, square kernels, symmetric stride/padding/dilation
            if child.groups != 1 or k_h != k_w or s_h != s_w or p_h != p_w or d_h != d_w:
                skipped += 1
                continue
            
            # This layer is convertible. Check if we should replace it.
            current_idx = counter_ctx["count"]
            counter_ctx["count"] += 1
            
            should_replace = True
            current_params = {}

            if target_idx is not None:
                if isinstance(target_idx, int):
                    if current_idx != target_idx:
                        should_replace = False
                elif isinstance(target_idx, (list, tuple, set)):
                    if current_idx not in target_idx:
                        should_replace = False
                elif isinstance(target_idx, dict):
                    if current_idx not in target_idx:
                        should_replace = False
                    else:
                        current_params = target_idx[current_idx]
            
            if not should_replace:
                continue

            mem_conv = Conv2dMem(
                current_params.get("engine", engine),
                child.in_channels,
                child.out_channels,
                k_h,
                input_slice=current_params.get("input_slice", input_slice),
                weight_slice=current_params.get("weight_slice", weight_slice),
                stride=s_h,
                padding=p_h,
                dilation=d_h,
                bias=(child.bias is not None),
                device=device,
                bw_e=current_params.get("bw_e", bw_e),
                input_paral_size=current_params.get("input_paral_size", input_paral_size),
                weight_paral_size=current_params.get("weight_paral_size", weight_paral_size),
                input_quant_gran=current_params.get("input_quant_gran", input_quant_gran),
                weight_quant_gran=current_params.get("weight_quant_gran", weight_quant_gran),
            )
            with torch.no_grad():
                mem_conv.weight.copy_(child.weight)
                if child.bias is not None and mem_conv.bias is not None:
                    mem_conv.bias.copy_(child.bias)
            setattr(module, name, mem_conv)
            converted += 1
        else:
            c = _replace_conv2d_with_mem(
                child,
                engine=engine,
                input_slice=input_slice,
                weight_slice=weight_slice,
                device=device,
                bw_e=bw_e,
                input_paral_size=input_paral_size,
                weight_paral_size=weight_paral_size,
                input_quant_gran=input_quant_gran,
                weight_quant_gran=weight_quant_gran,
                target_idx=target_idx,
                counter_ctx=counter_ctx,
            )
            converted += c["converted"]
            skipped += c["skipped"]

    return {"converted": converted, "skipped": skipped}


class YOLOv5(nn.Module):
    """
    YOLOv5 model wrapper loaded via torch.hub.

    Args:
        model_name: One of 'yolov5n', 'yolov5s', 'yolov5m'
        pretrained: Whether to load pretrained weights
        img_size: Inference image size (default 640)
        device: Torch device. If None, will use CUDA if available else CPU.
        hub_repo: Torch hub repo (default 'ultralytics/yolov5')
        hub_source: Torch hub source ('github' by default; can be 'local' if you mirror the repo)
        conf_thres: Confidence threshold (will set `model.conf` if provided)
        iou_thres: NMS IoU threshold (will set `model.iou` if provided)
        max_det: Max detections per image (will set `model.max_det` if provided)
        autoshape: If True, load autoshape model (accepts various input types). For evaluation pipelines
                  using torch tensors, `autoshape=False` is recommended for more direct control.

    Important:
        `mem_enabled` is accepted for API consistency but is currently not implemented for YOLOv5.
    """

    def __init__(
        self,
        model_name: str = "yolov5s",
        pretrained: bool = True,
        img_size: int = 640,
        device: Optional[torch.device] = None,
        # Pin to a stable YOLOv5 tag to avoid newer hubconf dependencies (e.g. requiring `ultralytics` package).
        hub_repo: str = "ultralytics/yolov5:v6.2",
        hub_source: str = "github",
        conf_thres: Optional[float] = None,
        iou_thres: Optional[float] = None,
        max_det: Optional[int] = None,
        # Use autoshape=True by default so the returned object is a Detections/Results-like
        # wrapper with `.pred` (used by our COCO evaluation example).
        autoshape: bool = True,
        mem_enabled: bool = False,
        mem_args: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()

        if model_name not in model_names:
            raise ValueError(f"Invalid model_name: {model_name}. Available: {list(model_names)}")

        self.model_name = model_name
        self.img_size = int(img_size)
        self.autoshape = bool(autoshape)
        self.mem_enabled = mem_enabled
        self.mem_args = mem_args if (mem_args is not None and mem_enabled) else {}

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # Load YOLOv5 via torch hub
        # NOTE: This will download code/weights to torch hub cache on first run.
        # PyTorch >=2.6 changed torch.load default `weights_only` from False -> True.
        # YOLOv5 official `.pt` checkpoints require `weights_only=False` to load.
        # We apply a temporary monkeypatch during hub loading.
        _orig_torch_load = torch.load

        def _torch_load_compat(*args, **kwargs):
            if "weights_only" not in kwargs:
                kwargs["weights_only"] = False
            return _orig_torch_load(*args, **kwargs)

        torch.load = _torch_load_compat  # type: ignore[assignment]
        try:
            try:
                self.model = torch.hub.load(
                    hub_repo,
                    model_name,
                    pretrained=pretrained,
                    autoshape=autoshape,
                    source=hub_source,
                    trust_repo=True,
                )
            except ModuleNotFoundError as e:
                # Newer YOLOv5 hubconf revisions may import `ultralytics` package.
                # If it's missing, fallback to a pinned YOLOv5 tag that does not require it.
                if str(e).strip() == "No module named 'ultralytics'":
                    fallback_repo = "ultralytics/yolov5:v6.1"
                    self.model = torch.hub.load(
                        fallback_repo,
                        model_name,
                        pretrained=pretrained,
                        autoshape=autoshape,
                        source=hub_source,
                        trust_repo=True,
                    )
                else:
                    raise
        finally:
            torch.load = _orig_torch_load  # type: ignore[assignment]
        self.model.to(self.device)

        # Convert conv/linear to memristive layers if enabled
        if self.mem_enabled:
            if not self.mem_args or self.mem_args.get("engine", None) is None:
                raise ValueError("mem_enabled=True 需要传入 mem_args['engine']=DPETensor(...)")

            # Determine the actual nn.Module to patch:
            # - AutoShape has attribute `.model` (DetectMultiBackend)
            # - DetectMultiBackend has attribute `.model` (PyTorch detection model when pt=True)
            patch_root = self.model
            if hasattr(patch_root, "model"):
                patch_root = getattr(patch_root, "model")  # AutoShape -> DetectMultiBackend
            if hasattr(patch_root, "model"):
                patch_root = getattr(patch_root, "model")  # DetectMultiBackend -> underlying pytorch model

            stats = _replace_conv2d_with_mem(
                patch_root,
                engine=self.mem_args["engine"],
                input_slice=self.mem_args.get("input_slice") or (1, 1, 2, 2),
                weight_slice=self.mem_args.get("weight_slice") or (1, 1, 2, 2),
                device=self.mem_args.get("device") or self.device,
                bw_e=self.mem_args.get("bw_e", None),
                input_paral_size=self.mem_args.get("input_paral_size") or (1, 32),
                weight_paral_size=self.mem_args.get("weight_paral_size") or (32, 32),
                input_quant_gran=self.mem_args.get("input_quant_gran") or (1, 64),
                weight_quant_gran=self.mem_args.get("weight_quant_gran") or (64, 64),
                target_idx=self.mem_args.get("target_idx", None),
            )
            self.mem_stats = stats
            # Keep a simple note for users
            if stats["skipped"] > 0:
                print(f"[MemIntelli][YOLOv5] 已将 {stats['converted']} 个 Conv2d 替换为 Conv2dMem，跳过 {stats['skipped']} 个不支持的 Conv2d（groups!=1 或非方形参数）。")
            else:
                print(f"[MemIntelli][YOLOv5] 已将 {stats['converted']} 个 Conv2d 替换为 Conv2dMem。")

        # Apply common inference params if provided
        if conf_thres is not None:
            self.model.conf = float(conf_thres)
        if iou_thres is not None:
            self.model.iou = float(iou_thres)
        if max_det is not None:
            self.model.max_det = int(max_det)

    def forward(self, x: torch.Tensor):
        """
        Forward pass.

        For torch-tensor inputs (B,3,H,W) in [0,1], YOLOv5 hub models can be called directly.
        Note: when `autoshape=False`, the underlying `DetectMultiBackend.forward()` does NOT accept `size=...`.
        """
        return self.model(x)

    def update_weight(self) -> None:
        """Update weights for memristive layers (if enabled)."""
        if not self.mem_enabled:
            return
        for m in self.modules():
            if isinstance(m, (Conv2dMem, LinearMem)):
                m.update_weight()


def YOLOv5_zoo(
    model_name: str = "yolov5s",
    pretrained: bool = True,
    img_size: int = 640,
    device: Optional[Any] = None,
    conf_thres: Optional[float] = None,
    iou_thres: Optional[float] = None,
    max_det: Optional[int] = None,
    autoshape: bool = True,
    mem_enabled: bool = False,
    engine: Optional[Any] = None,
    # Keep signature similar to other *_zoo APIs; currently unused for YOLOv5.
    input_slice: Optional[Any] = None,
    weight_slice: Optional[Any] = None,
    bw_e: Optional[Any] = None,
    input_paral_size: Optional[Any] = None,
    weight_paral_size: Optional[Any] = None,
    input_quant_gran: Optional[Any] = None,
    weight_quant_gran: Optional[Any] = None,
    replace_layer_idx: Optional[Union[int, List[int], Dict[int, Any]]] = None,
) -> YOLOv5:
    """
    YOLOv5 model factory.

    Supported pretrained variants:
    - 'yolov5n'
    - 'yolov5s'
    - 'yolov5m'
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mem_args = {
        "engine": engine,
        "input_slice": input_slice,
        "weight_slice": weight_slice,
        "device": device,
        "bw_e": bw_e,
        "input_paral_size": input_paral_size,
        "weight_paral_size": weight_paral_size,
        "input_quant_gran": input_quant_gran,
        "weight_quant_gran": weight_quant_gran,
        "target_idx": replace_layer_idx,
    }
    # Filter out None values so defaults in YOLOv5.__init__ can apply
    if mem_enabled:
        mem_args = {k: v for k, v in mem_args.items() if v is not None}
    else:
        mem_args = None

    return YOLOv5(
        model_name=model_name,
        pretrained=pretrained,
        img_size=img_size,
        device=device,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        max_det=max_det,
        autoshape=autoshape,
        mem_enabled=mem_enabled,
        mem_args=mem_args,
    )

