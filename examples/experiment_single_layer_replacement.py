# -*- coding:utf-8 -*-
# @File  : experiment_single_layer_replacement.py
# @Author: MemIntelli contributors
# @Date  : 2026/01/21
"""
Memintelli Experiment: Single Layer Replacement Test for YOLOv5.
Iteratively replaces one Conv2d layer at a time with Conv2dMem and evaluates mAP.
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
import urllib.request
import zipfile
import csv
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CocoDetection
from PIL import Image
from tqdm import tqdm

# Silence noisy warnings globally
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r"pkg_resources is deprecated as an API\.",
)

# Ensure we import the local MemIntelli source tree
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Import MemIntelli modules
from memintelli.NN_models.YOLOv5 import YOLOv5_zoo
from memintelli.pimpy.memmat_tensor import DPETensor
from memintelli.NN_layers import Conv2dMem

# --- Reusing helper functions from 11_yolov5_coco_inference.py ---

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
    img_root = data_root / "images" / "val2017"
    ann_file = data_root / "annotations" / "instances_val2017.json"
    return img_root, ann_file

def download_coco_val2017(data_root: Path) -> None:
    coco_img_zip = "http://images.cocodataset.org/zips/val2017.zip"
    coco_ann_zip = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

    dl_dir = data_root / "_downloads"
    val_zip_path = dl_dir / "val2017.zip"
    ann_zip_path = dl_dir / "annotations_trainval2017.zip"

    _download(coco_img_zip, val_zip_path)
    _download(coco_ann_zip, ann_zip_path)

    tmp_extract = dl_dir / "_extract"
    if tmp_extract.exists():
        pass

    _unzip(val_zip_path, tmp_extract)
    extracted_val = tmp_extract / "val2017"
    if not extracted_val.exists():
        raise RuntimeError(f"解压失败：未找到 {str(extracted_val)}")
    (data_root / "images").mkdir(parents=True, exist_ok=True)
    target_val = data_root / "images" / "val2017"
    if not target_val.exists():
        extracted_val.replace(target_val)

    _unzip(ann_zip_path, data_root)

class CocoVal640(Dataset):
    def __init__(self, img_root: str, ann_file: str, img_size: int = 640):
        _require("pycocotools", "pip install pycocotools")
        self.ds = CocoDetection(root=img_root, annFile=ann_file)
        self.img_size = int(img_size)
        cat_ids = sorted(self.ds.coco.getCatIds())
        self.cat_id_to_label = {cid: i for i, cid in enumerate(cat_ids)}

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        img, anns = self.ds[idx]
        w0, h0 = img.size
        scale = min(self.img_size / w0, self.img_size / h0)
        nw = int(w0 * scale)
        nh = int(h0 * scale)
        img_resized = img.resize((nw, nh), resample=Image.BILINEAR)

        new_img = Image.new("RGB", (self.img_size, self.img_size), (114, 114, 114))
        dx = (self.img_size - nw) // 2
        dy = (self.img_size - nh) // 2
        new_img.paste(img_resized, (dx, dy))
        img = new_img

        arr = np.array(img, dtype=np.uint8, copy=True)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        img_t = torch.from_numpy(arr).permute(2, 0, 1).contiguous().float() / 255.0

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
            x, y, bw, bh = bbox
            if bw <= 0 or bh <= 0:
                continue
            x1 = x * scale + dx
            y1 = y * scale + dy
            x2 = (x + bw) * scale + dx
            y2 = (y + bh) * scale + dy
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
) -> Dict[str, Any]:
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    _require("torchmetrics", "pip install torchmetrics")
    from torchmetrics.detection.mean_ap import MeanAveragePrecision

    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
    metric.warn_on_many_detections = False
    metric.to(device)

    model.eval()
    pbar = tqdm(loader, desc="Evaluating", unit="batch", leave=False)

    with torch.no_grad():
        for bi, (images, targets) in enumerate(pbar):
            images = images.to(device, non_blocking=True)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            results = model(images)
            preds: List[Dict[str, torch.Tensor]] = []

            if hasattr(results, "pred"):
                for det in results.pred:
                    if det is None or det.numel() == 0:
                        preds.append({
                            "boxes": torch.zeros((0, 4), dtype=torch.float32, device=device),
                            "scores": torch.zeros((0,), dtype=torch.float32, device=device),
                            "labels": torch.zeros((0,), dtype=torch.int64, device=device),
                        })
                    else:
                        preds.append({
                            "boxes": det[:, 0:4].float(),
                            "scores": det[:, 4].float(),
                            "labels": det[:, 5].long(),
                        })
            else:
                # Manual NMS for some raw outputs
                from utils.general import non_max_suppression # Assuming utils.general is available via sys.path if needed
                y = results
                if isinstance(y, (tuple, list)):
                    y = y[0]
                
                conf = getattr(getattr(model, "model", model), "conf", 0.001)
                iou = getattr(getattr(model, "model", model), "iou", 0.50)
                max_det = getattr(getattr(model, "model", model), "max_det", 300)
                dets = non_max_suppression(y, conf_thres=float(conf), iou_thres=float(iou), max_det=int(max_det), multi_label=True)
                for det in dets:
                    if det is None or det.numel() == 0:
                         preds.append({
                            "boxes": torch.zeros((0, 4), dtype=torch.float32, device=device),
                            "scores": torch.zeros((0,), dtype=torch.float32, device=device),
                            "labels": torch.zeros((0,), dtype=torch.int64, device=device),
                        })
                    else:
                        preds.append({
                            "boxes": det[:, 0:4].float(),
                            "scores": det[:, 4].float(),
                            "labels": det[:, 5].long(),
                        })

            metric.update(preds, targets)
            if max_batches and (bi + 1) >= max_batches:
                break
    
    return metric.compute()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="yolov5n", choices=["yolov5n", "yolov5s", "yolov5m"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data-root", type=str, default="")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--max-batches", type=int, default=5000)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--output-csv", type=str, default="layer_replacement_results.csv")
    parser.add_argument("--var", type=float, default=0.02, help="DPETensor variation variance parameter")
    parser.add_argument("--rvar", type=float, default=0.0, help="DPETensor variation variance parameter")
    parser.add_argument("--radc-bits", type=int, default=12, help="ADC 精度位数")
    
    args = parser.parse_args()
    
    data_root = Path(args.data_root) if args.data_root else Path("/data/dataset/coco")
    img_root, ann_file = maybe_prepare_coco_val2017(data_root)
    
    if not img_root.exists() or not ann_file.exists():
        if args.download:
            print("检测到 COCO val2017 数据缺失，开始自动下载...")
            data_root.mkdir(parents=True, exist_ok=True)
            download_coco_val2017(data_root)
        else:
             raise FileNotFoundError("COCO dataset missing. Use --download or specify valid --data-root.")

    device = torch.device(args.device)
    
    # Dataset only needs to be loaded once
    dataset = CocoVal640(str(img_root), str(ann_file), img_size=args.img_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=min(8, os.cpu_count() or 0),
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
    )
    
    print(f"Start Single Layer Replacement Experiment. Output: {args.output_csv}")
    
    # Create CSV and write header
    with open(args.output_csv, 'w', newline='') as csvfile:
        fieldnames = ['layer_idx', 'map_5095', 'map_50']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        layer_idx = 0
        while True:
            # Re-initialize model for each layer index
            print(f"\n--- Testing replacement of Layer Index {layer_idx} ---")
            
             # Slicing configuration
            def parse_slice(s):
                return tuple(map(int, s.split(',')))

            # Should be passed as args ideally, but hardcoding defaults for now as in orig script
            input_slice = (1, 1, 2, 4)
            weight_slice = (1, 1, 2, 4)

            mem_engine = DPETensor(
                HGS=1e-5,
                LGS=1e-8,
                rate_stuck_HGS=0.00,
                rate_stuck_LGS=0.00,
                read_variation=args.rvar,
                vnoise=0.0,
                write_variation=args.var,
                rdac=2**4,
                g_level=2**4,
                radc=2**args.radc_bits,
                device=device,
            )

            try:
                yolo = YOLOv5_zoo(
                    model_name=args.model,
                    pretrained=True,
                    img_size=args.img_size,
                    device=device,
                    conf_thres=args.conf,
                    iou_thres=args.iou,
                    max_det=args.max_det,
                    autoshape=True,
                    mem_enabled=True,
                    engine=mem_engine,
                    input_slice=input_slice,
                    weight_slice=weight_slice,
                    input_paral_size=(1, 64),
                    weight_paral_size=(64, 1),
                    input_quant_gran=(1, 64),
                    weight_quant_gran=(64, 1),
                    replace_layer_idx=layer_idx
                ).to(device)
            except Exception as e:
                print(f"Error initializing model: {e}")
                break

            # Force multi_label=True (same as in original example)
            if hasattr(yolo.model, "multi_label"):
                yolo.model.multi_label = True
            
            # Check if we successfully replaced the layer.
            # If converted == 0, it means layer_idx is out of range.
            if hasattr(yolo, 'mem_stats'):
                converted = yolo.mem_stats.get('converted', 0)
                if converted == 0:
                    print(f"Layer index {layer_idx} out of range (no layer converted). Stopping experiment.")
                    break
            else:
                # Should not happen if YOLOv5 code was patched correctly
                print("Warning: yolo.mem_stats missing. Assuming layer valid.")

            yolo.update_weight()
            
            stats = evaluate(yolo, loader, device, max_batches=args.max_batches)
            
            map_5095 = float(stats["map"].detach().cpu())
            map_50 = float(stats["map_50"].detach().cpu())
            
            print(f"Layer {layer_idx} | mAP50-95: {map_5095:.4f} | mAP50: {map_50:.4f}")
            
            writer.writerow({
                'layer_idx': layer_idx,
                'map_5095': map_5095,
                'map_50': map_50
            })
            csvfile.flush() # Ensure data is written
            
            layer_idx += 1

    print("Experiment completed.")

if __name__ == "__main__":
    main()
