# -*- coding:utf-8 -*-
# @File  : 11_yolov5_coco_inference.py
# @Author: MemIntelli contributors
# @Date  : 2026/01/15
"""
Memintelli example 11: YOLOv5 on COCO val2017 (mAP50-95 and mAP50).

Dataset layout (default under /data/dataset/coco):
  /data/dataset/coco/
    images/val2017/*.jpg
    annotations/instances_val2017.json

This script evaluates a pretrained YOLOv5n/s/m model at image size 640.
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CocoDetection
from PIL import Image
from tqdm import tqdm

# Silence noisy warnings globally (some are triggered during torch.hub import/load before evaluate()).
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r"pkg_resources is deprecated as an API\.",
)

# Ensure we import the local MemIntelli source tree (repo root) instead of an older site-packages install.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from memintelli.NN_models.YOLOv5 import YOLOv5_zoo
from memintelli.pimpy.memmat_tensor import DPETensor
from memintelli.NN_layers import Conv2dMem


def _require(pkg: str, install_hint: str) -> None:
    try:
        __import__(pkg)
    except Exception as e:
        raise RuntimeError(f"缺少依赖 {pkg}，请先安装：{install_hint}\n原始错误：{e}") from e


def _download(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        return
    print(f"下载中：{url}")
    print(f"保存到：{str(dst)}")
    urllib.request.urlretrieve(url, str(dst))


def _unzip(zip_path: Path, dst_dir: Path) -> None:
    print(f"解压中：{str(zip_path)} -> {str(dst_dir)}")
    dst_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(str(zip_path), "r") as zf:
        zf.extractall(str(dst_dir))


def maybe_prepare_coco_val2017(data_root: Path) -> Tuple[Path, Path]:
    """
    Ensure COCO val2017 exists under data_root:
      data_root/images/val2017
      data_root/annotations/instances_val2017.json
    If missing, download & extract when --download is used.
    """
    img_root = data_root / "images" / "val2017"
    ann_file = data_root / "annotations" / "instances_val2017.json"
    return img_root, ann_file


def download_coco_val2017(data_root: Path) -> None:
    """
    Download COCO val2017 images and annotations to data_root.
    Source: official COCO download links.
    """
    coco_img_zip = "http://images.cocodataset.org/zips/val2017.zip"
    coco_ann_zip = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

    dl_dir = data_root / "_downloads"
    val_zip_path = dl_dir / "val2017.zip"
    ann_zip_path = dl_dir / "annotations_trainval2017.zip"

    _download(coco_img_zip, val_zip_path)
    _download(coco_ann_zip, ann_zip_path)

    # Extract: val2017.zip contains val2017/; place under data_root/images/val2017
    tmp_extract = dl_dir / "_extract"
    if tmp_extract.exists():
        # keep it simple; reusing is ok
        pass

    _unzip(val_zip_path, tmp_extract)
    extracted_val = tmp_extract / "val2017"
    if not extracted_val.exists():
        raise RuntimeError(f"解压失败：未找到 {str(extracted_val)}")
    (data_root / "images").mkdir(parents=True, exist_ok=True)
    # Move/copy by rename when possible; if target exists, skip
    target_val = data_root / "images" / "val2017"
    if not target_val.exists():
        extracted_val.replace(target_val)

    # Extract annotations zip (contains annotations/instances_val2017.json etc.)
    _unzip(ann_zip_path, data_root)


class CocoVal640(Dataset):
    """
    COCO val2017 dataset wrapper:
    - resize image to (640, 640) (non-letterbox, direct resize)
    - scale bbox accordingly
    - map COCO category_id to contiguous 0..79 labels
    """

    def __init__(self, img_root: str, ann_file: str, img_size: int = 640):
        _require("pycocotools", "pip install pycocotools")
        self.ds = CocoDetection(root=img_root, annFile=ann_file)
        self.img_size = int(img_size)

        # COCO category ids are not contiguous; build mapping to 0..(K-1)
        cat_ids = sorted(self.ds.coco.getCatIds())
        self.cat_id_to_label = {cid: i for i, cid in enumerate(cat_ids)}

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        img, anns = self.ds[idx]
        w0, h0 = img.size  # PIL: (W, H)

        # Resize (direct resize, no letterbox)
        img = img.resize((self.img_size, self.img_size), resample=Image.BILINEAR)

        # ToTensor (float32, [0,1], (3,H,W))
        # np.asarray(PIL.Image) may yield a non-writable view; make a writable copy to avoid warnings.
        arr = np.array(img, dtype=np.uint8, copy=True)
        if arr.ndim == 2:  # grayscale -> 3ch
            arr = np.stack([arr, arr, arr], axis=-1)
        if arr.shape[-1] == 4:  # RGBA -> RGB
            arr = arr[..., :3]
        img_t = torch.from_numpy(arr).permute(2, 0, 1).contiguous().float() / 255.0

        sx = self.img_size / float(w0)
        sy = self.img_size / float(h0)

        boxes: List[List[float]] = []
        labels: List[int] = []

        for a in anns:
            if a.get("iscrowd", 0) == 1:
                continue
            bbox = a.get("bbox", None)
            cid = a.get("category_id", None)
            if bbox is None or cid is None:
                continue
            if cid not in self.cat_id_to_label:
                continue

            x, y, bw, bh = bbox  # COCO xywh
            if bw <= 0 or bh <= 0:
                continue

            x1 = x * sx
            y1 = y * sy
            x2 = (x + bw) * sx
            y2 = (y + bh) * sy

            # clamp
            x1 = max(0.0, min(float(self.img_size), x1))
            y1 = max(0.0, min(float(self.img_size), y1))
            x2 = max(0.0, min(float(self.img_size), x2))
            y2 = max(0.0, min(float(self.img_size), y2))
            if x2 <= x1 or y2 <= y1:
                continue

            boxes.append([x1, y1, x2, y2])
            labels.append(int(self.cat_id_to_label[cid]))

        if len(boxes) == 0:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes_t = torch.tensor(boxes, dtype=torch.float32)
            labels_t = torch.tensor(labels, dtype=torch.int64)

        target = {"boxes": boxes_t, "labels": labels_t}
        return img_t, target


def collate_fn(batch: List[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]):
    imgs, targets = zip(*batch)
    return torch.stack(list(imgs), dim=0), list(targets)


def evaluate(
    model,
    loader: DataLoader,
    device: torch.device,
    max_batches: int = 0,
    mem_debug: bool = False,
) -> Dict[str, Any]:
    # Silence noisy warnings (requested):
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message=r"pkg_resources is deprecated as an API\.",
    )
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message=r"Encountered more than 100 detections in a single image\.",
    )
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        message=r"`torch\.cuda\.amp\.autocast\(args\.\.\.\)` is deprecated\.",
    )

    _require("torchmetrics", "pip install torchmetrics")
    from torchmetrics.detection.mean_ap import MeanAveragePrecision  # type: ignore

    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
    # Disable torchmetrics internal warning about many detections
    metric.warn_on_many_detections = False
    metric.to(device)

    model.eval()
    pbar = tqdm(loader, desc="Evaluating (COCO val2017)", unit="batch")

    # Optional: verify MemIntelli path by hooking the first Conv2dMem
    hook_state = {"calls": 0}
    hook_handle = None
    if mem_debug:
        first_mem_conv = None
        for m in model.modules():
            if isinstance(m, Conv2dMem):
                first_mem_conv = m
                break
        if first_mem_conv is not None:
            def _hook(_module, _inp, _out):
                hook_state["calls"] += 1
            hook_handle = first_mem_conv.register_forward_hook(_hook)

    with torch.no_grad():
        for bi, (images, targets) in enumerate(pbar):
            images = images.to(device, non_blocking=True)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            results = model(images)

            preds: List[Dict[str, torch.Tensor]] = []
            # Case 1: results is a Detections-like object with `.pred` (list[tensor Nx6])
            if hasattr(results, "pred"):
                for det in results.pred:  # det: (N,6) xyxy, conf, cls
                    if det is None or det.numel() == 0:
                        preds.append(
                            {
                                "boxes": torch.zeros((0, 4), dtype=torch.float32, device=device),
                                "scores": torch.zeros((0,), dtype=torch.float32, device=device),
                                "labels": torch.zeros((0,), dtype=torch.int64, device=device),
                            }
                        )
                    else:
                        preds.append(
                            {
                                "boxes": det[:, 0:4].float(),
                                "scores": det[:, 4].float(),
                                "labels": det[:, 5].long(),
                            }
                        )
            else:
                # Case 2: results is raw model output tensor (B, N, 5+nc) requiring NMS
                # This happens in YOLOv5 v6.2 AutoShape() when input is a torch.Tensor.
                from utils.general import non_max_suppression  # type: ignore  # YOLOv5 repo util (available via torch.hub)

                y = results
                if isinstance(y, (tuple, list)):
                    y = y[0]
                if not isinstance(y, torch.Tensor):
                    raise TypeError(f"Unexpected YOLOv5 output type: {type(y)}")

                conf = getattr(getattr(model, "model", model), "conf", 0.001)
                iou = getattr(getattr(model, "model", model), "iou", 0.65)
                max_det = getattr(getattr(model, "model", model), "max_det", 300)
                dets = non_max_suppression(y, conf_thres=float(conf), iou_thres=float(iou), max_det=int(max_det))

                for det in dets:
                    if det is None or det.numel() == 0:
                        preds.append(
                            {
                                "boxes": torch.zeros((0, 4), dtype=torch.float32, device=device),
                                "scores": torch.zeros((0,), dtype=torch.float32, device=device),
                                "labels": torch.zeros((0,), dtype=torch.int64, device=device),
                            }
                        )
                    else:
                        preds.append(
                            {
                                "boxes": det[:, 0:4].float(),
                                "scores": det[:, 4].float(),
                                "labels": det[:, 5].long(),
                            }
                        )

            metric.update(preds, targets)

            if max_batches and (bi + 1) >= max_batches:
                break

    if hook_handle is not None:
        hook_handle.remove()
        print(f"[MemIntelli][Check] Conv2dMem forward hook calls: {hook_state['calls']} (应 > 0 表示确实走了存算层)")

    return metric.compute()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="yolov5n", choices=["yolov5n", "yolov5s", "yolov5m"])
    # batch-size in memristive mode is extremely memory hungry (especially at img=640),
    # so we decide the default dynamically after parsing.
    parser.add_argument("--batch-size", type=int, default=None, help="software 模式默认 8；mem 模式默认 1（更省显存）")
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data-root", type=str, default="")
    parser.add_argument("--download", action="store_true", help="若 COCO val2017 缺失则自动下载到 data-root")
    parser.add_argument("--max-batches", type=int, default=100, help=">0 时只评估前 N 个 batch（调试用）")
    parser.add_argument("--conf", type=float, default=0.001, help="YOLOv5 conf threshold")
    parser.add_argument("--iou", type=float, default=0.65, help="YOLOv5 NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=300, help="YOLOv5 max det per image")
    parser.add_argument("--mem-enabled", action="store_true", help="启用 MemIntelli（DPETensor）存算阵列模拟推理")
    parser.add_argument("--mem-debug", action="store_true", help="打印/验证是否走 MemIntelli 存算路径（Conv2dMem hook）")
    args = parser.parse_args()

    # Decide default batch size
    if args.batch_size is None:
        args.batch_size = 1 if args.mem_enabled else 1

    # Default to system-wide dataset directory (as requested): /data/dataset/coco
    data_root = Path(args.data_root) if args.data_root else Path("/data/dataset/coco")

    img_root, ann_file = maybe_prepare_coco_val2017(data_root)

    if not img_root.exists() or not ann_file.exists():
        if args.download:
            print("检测到 COCO val2017 数据缺失，开始自动下载...")
            data_root.mkdir(parents=True, exist_ok=True)
            download_coco_val2017(data_root)
        else:
            raise FileNotFoundError(
                "未找到 COCO val2017 数据集。\n"
                f"期望路径：\n- images: {str(img_root)}\n- ann: {str(ann_file)}\n\n"
                "解决方法：\n"
                "1) 手动下载并放到上述路径；或\n"
                "2) 直接加参数自动下载：--download\n"
                "也可以用 --data-root 指定 COCO 根目录。"
            )

    # Re-check after optional download
    if not img_root.exists() or not ann_file.exists():
        raise FileNotFoundError(f"COCO 数据仍缺失：images={str(img_root)} ann={str(ann_file)}")

    device = torch.device(args.device)

    # Slicing configuration and INT/FP mode settings (same style as Example 09)
    input_slice = (1, 1, 1, 2, 2)
    weight_slice = (1, 1, 1, 2, 2)
    bw_e = None

    mem_engine = None
    if args.mem_enabled:
        mem_engine = DPETensor(
            HGS=1e-5,
            LGS=1e-8,
            write_variation=0.05,
            rate_stuck_HGS=0.00,
            rate_stuck_LGS=0.00,
            read_variation=0.00,
            vnoise=0.0,
            rdac=2**2,
            g_level=2**2,
            radc=2**12,
            device=device,
        )

    # Load model
    yolo = YOLOv5_zoo(
        model_name=args.model,
        pretrained=True,
        img_size=args.img_size,
        device=device,
        conf_thres=args.conf,
        iou_thres=args.iou,
        max_det=args.max_det,
        autoshape=True,
        mem_enabled=bool(args.mem_enabled),
        engine=mem_engine,
        input_slice=input_slice,
        weight_slice=weight_slice,
        bw_e=bw_e,
        input_paral_size=(1, 64),
        weight_paral_size=(64, 1),
        input_quant_gran=(1, 64),
        weight_quant_gran=(64, 1),
    ).to(device)

    # print(yolo)

    if args.mem_enabled:
        # Convert original weights into quantized sliced weights for PIM simulation
        yolo.update_weight()
        # Basic checks
        mem_convs = [m for m in yolo.modules() if isinstance(m, Conv2dMem)]
        print(f"[MemIntelli][Check] Conv2dMem count: {len(mem_convs)}")
        if len(mem_convs) == 0:
            raise RuntimeError("mem_enabled=True 但模型中未检测到 Conv2dMem，未走存算路径。")
        if not isinstance(mem_convs[0].engine, DPETensor):
            raise RuntimeError(f"Conv2dMem.engine 不是 DPETensor，实际为：{type(mem_convs[0].engine)}")
        print(f"[MemIntelli][Check] First Conv2dMem.engine type: {type(mem_convs[0].engine)}")

    dataset = CocoVal640(str(img_root), str(ann_file), img_size=args.img_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=min(8, os.cpu_count() or 0),
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
    )

    stats = evaluate(yolo, loader, device, max_batches=args.max_batches, mem_debug=bool(args.mem_debug and args.mem_enabled))
    map_5095 = float(stats["map"].detach().cpu())
    map_50 = float(stats["map_50"].detach().cpu())
    print(f"\nCOCO val2017 @img{args.img_size} - {args.model}")
    print(f"mAP50-95: {map_5095:.4f}")
    print(f"mAP50   : {map_50:.4f}")


if __name__ == "__main__":
    main()

