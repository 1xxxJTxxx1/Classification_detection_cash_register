import os
import json
import copy
import random
import shutil
import argparse
import hashlib
from typing import List, Dict, Tuple, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageTk

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torchvision.transforms.functional as TF
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    f1_score,
    balanced_accuracy_score,
    accuracy_score,
    precision_score,
)

import tkinter as tk
from tkinter import ttk, filedialog, messagebox


# ============================================================
# НАСТРОЙКИ
# ============================================================

SEED = 42
PIPELINE_VERSION = "detector_classifier_v3"

DEFAULT_DATASET_PATH = "dataset"

OUTPUT_DIR = "cash_register_artifacts"
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")

META_PATH = os.path.join(MODEL_DIR, "model_meta.json")
TRAINING_SUMMARY_PATH = os.path.join(MODEL_DIR, "training_summary.json")
SPLIT_INFO_PATH = os.path.join(OUTPUT_DIR, "dataset_split.json")

DETECTOR_BEST_PATH = os.path.join(MODEL_DIR, "detector_best.pth")
DETECTOR_LAST_PATH = os.path.join(MODEL_DIR, "detector_last.pth")

DEMO_POOL_DIR = os.path.join(OUTPUT_DIR, "demo_images")
DEMO_META_PATH = os.path.join(DEMO_POOL_DIR, "demo_meta.json")
DEMO_AUG_COPIES_PER_IMAGE = 4
DEMO_RATIO = 0.22

CLASS_NAMES = {
    0: "закрыта",
    1: "открыта",
}
CLASS_FILE_TOKENS = {
    0: "closed",
    1: "open",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ---- classifier decision tuning ----
DEFAULT_OPEN_THRESHOLD = 0.58
UNCERTAINTY_MARGIN = 0.03

FALSE_OPEN_COST = 1.8
FALSE_CLOSED_COST = 1.0

# ---- training ----
N_SPLITS = 5
NUM_EPOCHS_CLASSIFIER = 32
NUM_EPOCHS_DETECTOR = 18

BATCH_SIZE_CLASSIFIER = 8
BATCH_SIZE_DETECTOR = 2

LEARNING_RATE_CLASSIFIER = 3e-4
LEARNING_RATE_DETECTOR = 3e-4

WEIGHT_DECAY_CLASSIFIER = 1e-4
WEIGHT_DECAY_DETECTOR = 5e-4

EARLY_STOPPING_PATIENCE = 7

FREEZE_BACKBONE = True
CLASSIFIER_ARCH = "efficientnet_b0"
CLASSIFIER_INPUT_SIZE = 224

# ---- detector inference ----
DETECTION_SCORE_THRESHOLD = 0.45
DETECTION_PAD_RATIO = 0.10

# ---- fallback ROI ----
# Он нужен только как резерв, если детектор не нашёл кассу.
# Сдвиг влево — по твоему наблюдению и по кадрам.
FALLBACK_ROI_WIDTH_PERCENTILE = 62
FALLBACK_ROI_HEIGHT_PERCENTILE = 68
FALLBACK_ROI_WIDTH_SCALE = 1.08
FALLBACK_ROI_HEIGHT_SCALE = 1.14
FALLBACK_CENTER_SHIFT_X_NORM = -0.016
FALLBACK_CENTER_SHIFT_Y_NORM = 0.010

# ---- candidate crops around detected box ----
LEFT_SHIFT_RATIO = 0.08
LEFT_EXPAND_RATIO = 0.12
RIGHT_EXPAND_RATIO = 0.02
DOWN_SHIFT_RATIO = 0.02

# ---- GUI ----
GUI_ORIGINAL_MAX_SIZE = (900, 500)
GUI_CROP_MAX_SIZE = (360, 260)

NUM_WORKERS = 0


def get_model_type_name() -> str:
    return f"FasterRCNN + {CLASSIFIER_ARCH}"


# ============================================================
# ОБЩИЕ ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cash register state detection/classification pipeline"
    )
    parser.add_argument(
        "--mode",
        choices=["train", "gui", "train_and_gui", "preview_dataset"],
        default="gui",
        help="train: train all stages; gui: run app using saved weights; train_and_gui: train then run app; preview_dataset: export preview images with boxes",
    )
    parser.add_argument(
        "--dataset-path",
        default=DEFAULT_DATASET_PATH,
        help="Path to dataset root with images, labels_detection, labels_classification",
    )
    parser.add_argument(
        "--force-train",
        action="store_true",
        help="Force retraining even if compatible best weights already exist",
    )
    parser.add_argument(
        "--rebuild-demo",
        action="store_true",
        help="Force regenerate demo pool from demo split",
    )
    parser.add_argument(
        "--preview-count",
        type=int,
        default=80,
        help="How many images to export in preview_dataset mode",
    )
    parser.add_argument(
        "--skip-gui",
        action="store_true",
        help="Backward-compatible alias: implies mode=train",
    )
    return parser.parse_args()


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA не обнаружена. Эта программа должна выполняться на видеокарте NVIDIA."
        )
    return torch.device("cuda")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def clear_dir(path: str) -> None:
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def list_image_files(images_dir: str) -> Dict[str, str]:
    result = {}
    for fname in os.listdir(images_dir):
        fpath = os.path.join(images_dir, fname)
        if not os.path.isfile(fpath):
            continue
        stem, ext = os.path.splitext(fname)
        if ext.lower() in IMAGE_EXTENSIONS:
            result[stem] = fpath
    return result


def list_txt_files(txt_dir: str) -> Dict[str, str]:
    result = {}
    for fname in os.listdir(txt_dir):
        fpath = os.path.join(txt_dir, fname)
        if not os.path.isfile(fpath):
            continue
        stem, ext = os.path.splitext(fname)
        if ext.lower() == ".txt":
            result[stem] = fpath
    return result


def safe_int_sort_key(text: str):
    try:
        return 0, int(text)
    except ValueError:
        return 1, text


def normalize_abs_path(path: str) -> str:
    return os.path.normcase(os.path.abspath(path))


def is_path_inside_dir(file_path: str, directory_path: str) -> bool:
    file_abs = normalize_abs_path(file_path)
    dir_abs = normalize_abs_path(directory_path)
    return file_abs == dir_abs or file_abs.startswith(dir_abs + os.sep)


def count_image_files_in_dir(directory: str) -> int:
    if not os.path.isdir(directory):
        return 0

    count = 0
    for fname in os.listdir(directory):
        _, ext = os.path.splitext(fname)
        if ext.lower() in IMAGE_EXTENSIONS:
            count += 1
    return count


def sha1_file(path: str) -> str:
    hasher = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def read_class_label(label_path: str) -> int:
    with open(label_path, "r", encoding="utf-8") as f:
        label = int(f.readline().strip())

    if label not in CLASS_NAMES:
        raise ValueError(f"Некорректная метка класса {label} в файле {label_path}")

    return label


def read_boxes_txt(label_path: str) -> np.ndarray:
    boxes = []

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue

            if len(parts) != 4:
                raise ValueError(
                    f"Файл {label_path} должен содержать строки формата: x1 y1 x2 y2"
                )

            x1, y1, x2, y2 = map(float, parts)

            if x2 < x1:
                x1, x2 = x2, x1
            if y2 < y1:
                y1, y2 = y2, y1

            boxes.append([x1, y1, x2, y2])

    if len(boxes) == 0:
        raise ValueError(f"В файле {label_path} нет ни одного bounding box")

    return np.array(boxes, dtype=np.float32)


def choose_largest_box(boxes: np.ndarray) -> np.ndarray:
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    idx = int(np.argmax(areas))
    return boxes[idx]


def build_samples(dataset_root: str) -> List[Dict]:
    images_dir = os.path.join(dataset_root, "images")
    det_dir = os.path.join(dataset_root, "labels_detection")
    cls_dir = os.path.join(dataset_root, "labels_classification")

    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"Не найдена папка: {images_dir}")
    if not os.path.isdir(det_dir):
        raise FileNotFoundError(f"Не найдена папка: {det_dir}")
    if not os.path.isdir(cls_dir):
        raise FileNotFoundError(f"Не найдена папка: {cls_dir}")

    image_files = list_image_files(images_dir)
    det_files = list_txt_files(det_dir)
    cls_files = list_txt_files(cls_dir)

    common_stems = sorted(
        set(image_files.keys()) & set(det_files.keys()) & set(cls_files.keys()),
        key=safe_int_sort_key
    )

    if not common_stems:
        raise RuntimeError(
            "Не найдено ни одного полного набора image + labels_detection + labels_classification"
        )

    samples = []
    for stem in common_stems:
        samples.append({
            "stem": stem,
            "image_path": image_files[stem],
            "det_label_path": det_files[stem],
            "cls_label_path": cls_files[stem],
            "class_label": read_class_label(cls_files[stem]),
        })

    return samples


def split_samples_for_detection(
    samples: List[Dict],
    val_ratio: float = 0.2,
    seed: int = SEED
) -> Tuple[List[Dict], List[Dict]]:
    if len(samples) < 4:
        raise RuntimeError(
            "Слишком мало train-данных для обучения детектора: нужно минимум 4 изображения."
        )

    labels = [sample["class_label"] for sample in samples]
    class_counts = {0: labels.count(0), 1: labels.count(1)}
    indices = np.arange(len(samples))

    stratify_labels = labels
    # Fallback for tiny/imbalanced subsets where stratified split can fail.
    if min(class_counts.values()) < 2:
        stratify_labels = None

    try:
        train_idx, val_idx = train_test_split(
            indices,
            test_size=val_ratio,
            random_state=seed,
            shuffle=True,
            stratify=stratify_labels
        )
    except ValueError:
        train_idx, val_idx = train_test_split(
            indices,
            test_size=val_ratio,
            random_state=seed,
            shuffle=True,
            stratify=None
        )

    train_samples = [samples[i] for i in train_idx]
    val_samples = [samples[i] for i in val_idx]
    return train_samples, val_samples


# ============================================================
# РАЗДЕЛЕНИЕ НА TRAIN / DEMO
# ============================================================

def compute_dataset_signature(samples: List[Dict]) -> str:
    parts = []
    for sample in sorted(samples, key=lambda s: safe_int_sort_key(s["stem"])):
        image_path = sample["image_path"]
        det_label_path = sample["det_label_path"]
        cls_label_path = sample["cls_label_path"]
        class_label = int(sample["class_label"])

        image_size = os.path.getsize(image_path)
        image_mtime_ns = os.stat(image_path).st_mtime_ns
        det_mtime_ns = os.stat(det_label_path).st_mtime_ns
        cls_mtime_ns = os.stat(cls_label_path).st_mtime_ns

        parts.append(
            f"{sample['stem']}|{class_label}|{image_size}|{image_mtime_ns}|{det_mtime_ns}|{cls_mtime_ns}"
        )

    payload = "\n".join(parts).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


def save_split_info(
    split_info_path: str,
    train_samples: List[Dict],
    demo_samples: List[Dict],
    dataset_signature: str,
) -> None:
    ensure_dir(os.path.dirname(split_info_path))
    data = {
        "pipeline_version": PIPELINE_VERSION,
        "dataset_signature": dataset_signature,
        "train_stems": [sample["stem"] for sample in train_samples],
        "demo_stems": [sample["stem"] for sample in demo_samples],
    }
    with open(split_info_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_split_info(split_info_path: str) -> Dict:
    with open(split_info_path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_or_load_demo_split(
    samples: List[Dict],
    split_info_path: str,
    demo_ratio: float = DEMO_RATIO
) -> Tuple[List[Dict], List[Dict], bool]:
    sample_by_stem = {sample["stem"]: sample for sample in samples}
    current_signature = compute_dataset_signature(samples)

    if os.path.exists(split_info_path):
        try:
            split_info = load_split_info(split_info_path)
            if (
                split_info.get("pipeline_version") == PIPELINE_VERSION
                and split_info.get("dataset_signature") == current_signature
            ):
                train_stems = split_info.get("train_stems", [])
                demo_stems = split_info.get("demo_stems", [])

                if train_stems and demo_stems:
                    saved_stems = train_stems + demo_stems
                    saved_set = set(saved_stems)
                    current_set = set(sample_by_stem.keys())
                    has_duplicates = len(saved_set) != len(saved_stems)

                    # Reuse split only if it matches the current dataset exactly.
                    if not has_duplicates and saved_set == current_set and set(train_stems).isdisjoint(demo_stems):
                        train_samples = [sample_by_stem[stem] for stem in train_stems]
                        demo_samples = [sample_by_stem[stem] for stem in demo_stems]

                        train_labels = [sample["class_label"] for sample in train_samples]
                        demo_labels = [sample["class_label"] for sample in demo_samples]

                        if len(set(train_labels)) == 2 and len(set(demo_labels)) == 2:
                            return train_samples, demo_samples, False
        except Exception:
            pass

    labels = [sample["class_label"] for sample in samples]
    indices = np.arange(len(samples))

    train_idx, demo_idx = train_test_split(
        indices,
        test_size=demo_ratio,
        random_state=SEED,
        shuffle=True,
        stratify=labels
    )

    train_samples = [samples[i] for i in train_idx]
    demo_samples = [samples[i] for i in demo_idx]

    save_split_info(
        split_info_path=split_info_path,
        train_samples=train_samples,
        demo_samples=demo_samples,
        dataset_signature=current_signature,
    )
    return train_samples, demo_samples, True


def validate_train_demo_disjoint(train_samples: List[Dict], demo_samples: List[Dict]) -> None:
    train_stems = {sample["stem"] for sample in train_samples}
    demo_stems = {sample["stem"] for sample in demo_samples}

    overlap_stems = train_stems & demo_stems
    if overlap_stems:
        raise RuntimeError(
            f"Train/demo overlap detected for stems: {sorted(list(overlap_stems))[:5]}"
        )

    train_hashes = {sha1_file(sample["image_path"]) for sample in train_samples}
    demo_hashes = {sha1_file(sample["image_path"]) for sample in demo_samples}
    overlap_hashes = train_hashes & demo_hashes
    if overlap_hashes:
        raise RuntimeError("Train/demo overlap detected by image content hashes.")


# ============================================================
# FALLBACK ROI
# ============================================================

def compute_fallback_roi_from_labels(
    samples: List[Dict]
) -> Tuple[Tuple[float, float, float, float], Tuple[int, int]]:
    """
    Резервный ROI, если детектор не сработал.
    Считается как медианный центр + робастный размер.
    """
    if not samples:
        raise ValueError("Список samples пуст")

    cx_list, cy_list = [], []
    bw_list, bh_list = [], []
    widths, heights = [], []

    for sample in samples:
        image = Image.open(sample["image_path"]).convert("RGB")
        width, height = image.size
        widths.append(width)
        heights.append(height)

        boxes = read_boxes_txt(sample["det_label_path"])
        best_box = choose_largest_box(boxes)

        x1, y1, x2, y2 = best_box
        bw = (x2 - x1) / width
        bh = (y2 - y1) / height
        cx = ((x1 + x2) / 2.0) / width
        cy = ((y1 + y2) / 2.0) / height

        bw_list.append(bw)
        bh_list.append(bh)
        cx_list.append(cx)
        cy_list.append(cy)

    cx_med = float(np.median(cx_list)) + FALLBACK_CENTER_SHIFT_X_NORM
    cy_med = float(np.median(cy_list)) + FALLBACK_CENTER_SHIFT_Y_NORM

    bw_robust = float(np.percentile(bw_list, FALLBACK_ROI_WIDTH_PERCENTILE)) * FALLBACK_ROI_WIDTH_SCALE
    bh_robust = float(np.percentile(bh_list, FALLBACK_ROI_HEIGHT_PERCENTILE)) * FALLBACK_ROI_HEIGHT_SCALE

    x1n = max(0.0, cx_med - bw_robust / 2.0)
    y1n = max(0.0, cy_med - bh_robust / 2.0)
    x2n = min(1.0, cx_med + bw_robust / 2.0)
    y2n = min(1.0, cy_med + bh_robust / 2.0)

    ref_width = int(round(np.median(widths)))
    ref_height = int(round(np.median(heights)))

    return (x1n, y1n, x2n, y2n), (ref_width, ref_height)


def normalized_roi_to_pixels(
    roi_norm: Tuple[float, float, float, float],
    width: int,
    height: int
) -> Tuple[int, int, int, int]:
    x1n, y1n, x2n, y2n = roi_norm

    x1 = max(0, min(width - 1, int(round(x1n * width))))
    y1 = max(0, min(height - 1, int(round(y1n * height))))
    x2 = max(x1 + 1, min(width, int(round(x2n * width))))
    y2 = max(y1 + 1, min(height, int(round(y2n * height))))

    return x1, y1, x2, y2


# ============================================================
# CROP HELPERS
# ============================================================

def clamp_box_to_image(
    box: Tuple[float, float, float, float],
    image_w: int,
    image_h: int
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    x1 = max(0, min(image_w - 1, int(round(x1))))
    y1 = max(0, min(image_h - 1, int(round(y1))))
    x2 = max(x1 + 1, min(image_w, int(round(x2))))
    y2 = max(y1 + 1, min(image_h, int(round(y2))))
    return x1, y1, x2, y2


def expand_box(
    box: Tuple[float, float, float, float],
    image_w: int,
    image_h: int,
    pad_ratio: float = 0.10
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = map(float, box)
    bw = x2 - x1
    bh = y2 - y1

    pad_x = bw * pad_ratio
    pad_y = bh * pad_ratio

    return clamp_box_to_image(
        (x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y),
        image_w,
        image_h
    )


def crop_by_box(image: Image.Image, box: Tuple[int, int, int, int]) -> Image.Image:
    return image.crop(box)


def make_inference_candidate_boxes(
    base_box: Tuple[int, int, int, int],
    image_w: int,
    image_h: int
) -> List[Tuple[Tuple[int, int, int, int], float]]:
    x1, y1, x2, y2 = map(float, base_box)
    bw = x2 - x1
    bh = y2 - y1

    # 1) центр
    box_center = clamp_box_to_image((x1, y1, x2, y2), image_w, image_h)

    # 2) сдвиг влево
    dx = bw * LEFT_SHIFT_RATIO
    box_left = clamp_box_to_image((x1 - dx, y1, x2 - dx, y2), image_w, image_h)

    # 3) расширение влево
    box_left_expand = clamp_box_to_image(
        (x1 - bw * LEFT_EXPAND_RATIO, y1, x2 + bw * RIGHT_EXPAND_RATIO, y2),
        image_w,
        image_h
    )

    # 4) чуть вниз
    dy = bh * DOWN_SHIFT_RATIO
    box_down = clamp_box_to_image((x1, y1 + dy, x2, y2 + dy), image_w, image_h)

    return [
        (box_center, 0.40),
        (box_left, 0.25),
        (box_left_expand, 0.25),
        (box_down, 0.10),
    ]


# ============================================================
# RESIZE С СОХРАНЕНИЕМ ПРОПОРЦИЙ
# ============================================================

class ResizePadSquare:
    def __init__(self, size: int, fill=(114, 114, 114)):
        self.size = size
        self.fill = fill

    def __call__(self, image: Image.Image) -> Image.Image:
        if image.mode != "RGB":
            image = image.convert("RGB")

        w, h = image.size
        if w <= 0 or h <= 0:
            raise ValueError("Некорректный размер изображения")

        scale = self.size / max(w, h)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))

        resized = image.resize((new_w, new_h), Image.LANCZOS)
        canvas = Image.new("RGB", (self.size, self.size), self.fill)

        left = (self.size - new_w) // 2
        top = (self.size - new_h) // 2
        canvas.paste(resized, (left, top))

        return canvas


# ============================================================
# ТРАНСФОРМАЦИИ
# ============================================================

class DetectionTrainTransform:
    """
    Для fixed-camera задачи держим аугментации мягкими.
    Без horizontal flip — он здесь скорее ломает семантику сцены.
    """
    def __init__(self):
        self.color = transforms.ColorJitter(
            brightness=0.05,
            contrast=0.05,
            saturation=0.03,
            hue=0.01
        )

    def __call__(self, image: Image.Image, target: Dict):
        if random.random() < 0.5:
            image = self.color(image)
        image = TF.to_tensor(image)
        return image, target


class DetectionEvalTransform:
    def __call__(self, image: Image.Image, target: Dict):
        image = TF.to_tensor(image)
        return image, target


def get_classification_train_transform():
    return transforms.Compose([
        transforms.RandomPerspective(distortion_scale=0.04, p=0.08),
        transforms.ColorJitter(
            brightness=0.08,
            contrast=0.08,
            saturation=0.06,
            hue=0.01
        ),
        ResizePadSquare(CLASSIFIER_INPUT_SIZE),
        transforms.ToTensor(),
        transforms.RandomErasing(
            p=0.14,
            scale=(0.015, 0.06),
            ratio=(0.5, 2.0),
            value="random"
        ),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


def get_classification_eval_transform():
    return transforms.Compose([
        ResizePadSquare(CLASSIFIER_INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


def get_demo_full_image_aug_transform():
    """
    Demo-аугментация мягче тренировочной.
    Только лёгкое визуальное разнообразие.
    """
    return transforms.Compose([
        transforms.ColorJitter(
            brightness=0.012,
            contrast=0.012,
            saturation=0.008,
            hue=0.002
        ),
        transforms.RandomAutocontrast(p=0.02),
        transforms.RandomAdjustSharpness(sharpness_factor=1.02, p=0.02),
    ])


# ============================================================
# DATASETS
# ============================================================

class DetectionDataset(Dataset):
    def __init__(self, samples: List[Dict], transform=None):
        self.samples = samples
        self.transform = transform if transform is not None else DetectionEvalTransform()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]

        image = Image.open(sample["image_path"]).convert("RGB")
        boxes_np = read_boxes_txt(sample["det_label_path"])
        boxes = torch.as_tensor(boxes_np, dtype=torch.float32)

        labels = torch.ones((boxes.shape[0],), dtype=torch.int64)
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "area": area,
            "iscrowd": iscrowd,
        }

        image, target = self.transform(image, target)
        return image, target


class ClassificationBoxCropDataset(Dataset):
    def __init__(
        self,
        samples: List[Dict],
        transform=None,
        pad_ratio: float = DETECTION_PAD_RATIO,
    ):
        self.samples = samples
        self.transform = transform if transform is not None else get_classification_eval_transform()
        self.pad_ratio = pad_ratio

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]

        image = Image.open(sample["image_path"]).convert("RGB")
        image_w, image_h = image.size

        boxes = read_boxes_txt(sample["det_label_path"])
        best_box = choose_largest_box(boxes)
        box = expand_box(tuple(best_box.tolist()), image_w, image_h, pad_ratio=self.pad_ratio)

        crop = crop_by_box(image, box)
        crop_tensor = self.transform(crop)

        label = int(sample["class_label"])
        return crop_tensor, label


def detection_collate_fn(batch):
    return tuple(zip(*batch))


# ============================================================
# МОДЕЛИ
# ============================================================

def build_detection_model(num_classes: int = 2, pretrained: bool = True):
    try:
        from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
        model = fasterrcnn_resnet50_fpn(weights=weights)
    except Exception:
        model = fasterrcnn_resnet50_fpn(pretrained=pretrained)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def build_classification_model(
    num_classes: int = 2,
    pretrained: bool = True,
    arch: str = CLASSIFIER_ARCH,
):
    arch = arch.lower().strip()

    if arch == "efficientnet_b0":
        try:
            from torchvision.models import EfficientNet_B0_Weights

            weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
            model = models.efficientnet_b0(weights=weights)
        except Exception:
            model = models.efficientnet_b0(pretrained=pretrained)

        in_features = model.classifier[-1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.30),
            nn.Linear(in_features, num_classes),
        )
    elif arch == "resnet34":
        try:
            from torchvision.models import ResNet34_Weights

            weights = ResNet34_Weights.DEFAULT if pretrained else None
            model = models.resnet34(weights=weights)
        except Exception:
            model = models.resnet34(pretrained=pretrained)

        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.28),
            nn.Linear(in_features, num_classes),
        )
    else:
        raise ValueError(f"Unsupported classifier arch: {arch}")

    if FREEZE_BACKBONE:
        for param in model.parameters():
            param.requires_grad = False

        for name, param in model.named_parameters():
            name = name.lower()
            if arch == "efficientnet_b0":
                if name.startswith("features.5") or name.startswith("features.6") or name.startswith("features.7") or name.startswith("classifier"):
                    param.requires_grad = True
            elif arch == "resnet34":
                if name.startswith("layer3") or name.startswith("layer4") or name.startswith("fc"):
                    param.requires_grad = True

    return model


# ============================================================
# DETECTION EVAL
# ============================================================

def evaluate_detection_loss(model, data_loader, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    count = 0

    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            total_loss += float(losses.item())
            count += 1

    model.eval()
    return total_loss / max(count, 1)


# ============================================================
# CLASSIFICATION EVAL / THRESHOLD TUNING
# ============================================================

def collect_probabilities(model, data_loader, criterion, device: torch.device):
    model.eval()

    total_loss = 0.0
    all_targets = []
    all_prob_open = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, labels)
            probs = torch.softmax(logits, dim=1)[:, 1]

            total_loss += float(loss.item())
            all_targets.extend(labels.cpu().numpy().tolist())
            all_prob_open.extend(probs.cpu().numpy().tolist())

    avg_loss = total_loss / max(len(data_loader), 1)
    all_targets = np.array(all_targets, dtype=np.int64)
    all_prob_open = np.array(all_prob_open, dtype=np.float32)

    return avg_loss, all_targets, all_prob_open


def _confusion_counts_binary(targets: np.ndarray, preds: np.ndarray) -> Tuple[int, int, int, int]:
    tn = int(np.sum((targets == 0) & (preds == 0)))
    fp = int(np.sum((targets == 0) & (preds == 1)))
    fn = int(np.sum((targets == 1) & (preds == 0)))
    tp = int(np.sum((targets == 1) & (preds == 1)))
    return tn, fp, fn, tp


def find_best_open_threshold_cost_sensitive(
    targets: np.ndarray,
    prob_open: np.ndarray
) -> Tuple[float, float, float, float, float]:
    thresholds = np.linspace(0.45, 0.75, 61)

    best_threshold = DEFAULT_OPEN_THRESHOLD
    best_cost = float("inf")
    best_acc = 0.0
    best_bal_acc = 0.0
    best_f1 = 0.0

    for threshold in thresholds:
        preds = (prob_open >= threshold).astype(np.int64)

        tn, fp, fn, tp = _confusion_counts_binary(targets, preds)
        cost = FALSE_OPEN_COST * fp + FALSE_CLOSED_COST * fn

        bal_acc = balanced_accuracy_score(targets, preds)
        acc = accuracy_score(targets, preds)
        f1 = f1_score(targets, preds, average="binary", zero_division=0)
        precision_open = precision_score(targets, preds, pos_label=1, zero_division=0)

        better = False
        if cost < best_cost:
            better = True
        elif cost == best_cost:
            if bal_acc > best_bal_acc:
                better = True
            elif bal_acc == best_bal_acc:
                current_best_preds = (prob_open >= best_threshold).astype(np.int64)
                current_best_precision_open = precision_score(
                    targets,
                    current_best_preds,
                    pos_label=1,
                    zero_division=0
                )
                if precision_open > current_best_precision_open:
                    better = True
                elif precision_open == current_best_precision_open and threshold > best_threshold:
                    better = True

        if better:
            best_threshold = float(threshold)
            best_cost = float(cost)
            best_bal_acc = float(bal_acc)
            best_acc = float(acc)
            best_f1 = float(f1)

    return best_threshold, best_acc, best_bal_acc, best_f1, best_cost


# ============================================================
# СЕРИАЛИЗАЦИЯ
# ============================================================

def save_checkpoint(
    save_path: str,
    model,
    optimizer=None,
    epoch: int = 0,
    best_metric: Optional[float] = None,
    extra: Optional[Dict] = None
):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "best_metric": best_metric,
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    if extra is not None:
        checkpoint.update(extra)

    torch.save(checkpoint, save_path)


def save_meta(
    meta_path: str,
    fallback_roi_norm: Tuple[float, float, float, float],
    reference_size: Tuple[int, int],
    open_threshold: float,
    train_count: int,
    demo_count: int,
    dataset_path: Optional[str] = None,
    dataset_signature: Optional[str] = None,
):
    ensure_dir(os.path.dirname(meta_path))

    meta = {
        "pipeline_version": PIPELINE_VERSION,
        "fallback_roi_norm": list(fallback_roi_norm),
        "reference_size": list(reference_size),
        "class_names": {str(k): v for k, v in CLASS_NAMES.items()},
        "open_threshold": open_threshold,
        "uncertainty_margin": UNCERTAINTY_MARGIN,
        "false_open_cost": FALSE_OPEN_COST,
        "false_closed_cost": FALSE_CLOSED_COST,
        "detection_score_threshold": DETECTION_SCORE_THRESHOLD,
        "model_type": get_model_type_name(),
        "classifier_arch": CLASSIFIER_ARCH,
        "classifier_input_size": CLASSIFIER_INPUT_SIZE,
        "dataset_path": dataset_path,
        "dataset_signature": dataset_signature,
        "num_classes": 2,
        "train_count": train_count,
        "demo_count": demo_count,
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def load_meta(meta_path: str) -> Dict:
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_classifier_fold_checkpoint_paths(model_dir: str) -> List[str]:
    if not os.path.isdir(model_dir):
        return []

    result = []
    for fname in sorted(os.listdir(model_dir)):
        if fname.startswith("classifier_fold_") and fname.endswith("_best.pth"):
            result.append(os.path.join(model_dir, fname))
    return result


def has_ready_models(
    model_dir: str,
    meta_path: str,
    expected_dataset_signature: Optional[str] = None
) -> bool:
    if not os.path.exists(meta_path):
        return False

    if not os.path.exists(DETECTOR_BEST_PATH):
        return False

    classifier_paths = get_classifier_fold_checkpoint_paths(model_dir)
    if len(classifier_paths) < 2:
        return False

    try:
        meta = load_meta(meta_path)
        if meta.get("pipeline_version") != PIPELINE_VERSION:
            return False
        if str(meta.get("classifier_arch", "")).lower() != str(CLASSIFIER_ARCH).lower():
            return False
        if int(meta.get("classifier_input_size", -1)) != int(CLASSIFIER_INPUT_SIZE):
            return False
        if expected_dataset_signature is not None:
            if meta.get("dataset_signature") != expected_dataset_signature:
                return False
    except Exception:
        return False

    return True


# ============================================================
# DEMO IMAGES
# ============================================================

def generate_demo_pool(
    train_samples: List[Dict],
    demo_samples: List[Dict],
    demo_pool_dir: str = DEMO_POOL_DIR,
    demo_meta_path: str = DEMO_META_PATH,
    dataset_signature: Optional[str] = None,
    copies_per_image: int = DEMO_AUG_COPIES_PER_IMAGE,
    seed: Optional[int] = SEED
) -> None:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    clear_dir(demo_pool_dir)
    aug_transform = get_demo_full_image_aug_transform()

    manifest = []

    for sample in demo_samples:
        image = Image.open(sample["image_path"]).convert("RGB")
        stem = sample["stem"]
        class_id = int(sample["class_label"])
        class_name = CLASS_NAMES[class_id]
        class_token = CLASS_FILE_TOKENS.get(class_id, f"class_{class_id}")

        original_name = f"{stem}_class_{class_id}_{class_token}_original.png"
        original_path = os.path.join(demo_pool_dir, original_name)
        image.save(original_path)

        manifest.append({
            "file_name": original_name,
            "source_stem": stem,
            "class_id": class_id,
            "class_name": class_name,
            "type": "original"
        })

        for i in range(1, copies_per_image + 1):
            aug_image = aug_transform(copy.deepcopy(image))
            aug_name = f"{stem}_class_{class_id}_{class_token}_aug_{i:02d}.png"
            aug_path = os.path.join(demo_pool_dir, aug_name)
            aug_image.save(aug_path)

            manifest.append({
                "file_name": aug_name,
                "source_stem": stem,
                "class_id": class_id,
                "class_name": class_name,
                "type": "augmented"
            })

    manifest_path = os.path.join(demo_pool_dir, "demo_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({
            "pipeline_version": PIPELINE_VERSION,
            "copies_per_image": copies_per_image,
            "items": manifest
        }, f, ensure_ascii=False, indent=2)

    with open(demo_meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "pipeline_version": PIPELINE_VERSION,
            "dataset_signature": dataset_signature,
            "copies_per_image": copies_per_image,
            "train_stems": [sample["stem"] for sample in train_samples],
            "demo_stems": [sample["stem"] for sample in demo_samples],
        }, f, ensure_ascii=False, indent=2)


def regenerate_demo_pool_from_saved_split(
    dataset_path: str,
    split_info_path: str = SPLIT_INFO_PATH,
    demo_pool_dir: str = DEMO_POOL_DIR,
    demo_meta_path: str = DEMO_META_PATH,
    copies_per_image: int = DEMO_AUG_COPIES_PER_IMAGE,
    seed: Optional[int] = None
) -> None:
    samples = build_samples(dataset_path)
    dataset_signature = compute_dataset_signature(samples)
    train_samples, demo_samples, _ = create_or_load_demo_split(
        samples=samples,
        split_info_path=split_info_path,
        demo_ratio=DEMO_RATIO
    )

    generate_demo_pool(
        train_samples=train_samples,
        demo_samples=demo_samples,
        demo_pool_dir=demo_pool_dir,
        demo_meta_path=demo_meta_path,
        dataset_signature=dataset_signature,
        copies_per_image=copies_per_image,
        seed=seed
    )


def is_demo_pool_up_to_date(
    demo_pool_dir: str,
    demo_meta_path: str,
    dataset_signature: str,
    copies_per_image: int
) -> bool:
    if not os.path.exists(demo_meta_path):
        return False
    if count_image_files_in_dir(demo_pool_dir) == 0:
        return False

    try:
        with open(demo_meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception:
        return False

    if meta.get("pipeline_version") != PIPELINE_VERSION:
        return False
    if meta.get("dataset_signature") != dataset_signature:
        return False
    if int(meta.get("copies_per_image", -1)) != int(copies_per_image):
        return False
    return True


def ensure_demo_pool(
    train_samples: List[Dict],
    demo_samples: List[Dict],
    dataset_signature: str,
    force_rebuild: bool = False,
    demo_pool_dir: str = DEMO_POOL_DIR,
    demo_meta_path: str = DEMO_META_PATH,
    copies_per_image: int = DEMO_AUG_COPIES_PER_IMAGE,
    seed: Optional[int] = SEED
) -> bool:
    up_to_date = is_demo_pool_up_to_date(
        demo_pool_dir=demo_pool_dir,
        demo_meta_path=demo_meta_path,
        dataset_signature=dataset_signature,
        copies_per_image=copies_per_image,
    )
    if up_to_date and not force_rebuild:
        return False

    generate_demo_pool(
        train_samples=train_samples,
        demo_samples=demo_samples,
        demo_pool_dir=demo_pool_dir,
        demo_meta_path=demo_meta_path,
        dataset_signature=dataset_signature,
        copies_per_image=copies_per_image,
        seed=seed,
    )
    return True


def export_dataset_preview(
    samples: List[Dict],
    output_dir: str,
    max_items: int = 80,
    seed: int = SEED
) -> int:
    clear_dir(output_dir)
    rng = random.Random(seed)

    indices = list(range(len(samples)))
    rng.shuffle(indices)
    indices = indices[: max(0, int(max_items))]

    exported = 0
    for idx in indices:
        sample = samples[idx]
        image = Image.open(sample["image_path"]).convert("RGB")
        boxes = read_boxes_txt(sample["det_label_path"])
        box = tuple(map(float, choose_largest_box(boxes).tolist()))
        roi = clamp_box_to_image(box, image.size[0], image.size[1])

        preview = draw_roi_on_image(image, roi)
        draw = ImageDraw.Draw(preview)
        class_name = CLASS_NAMES[int(sample["class_label"])]
        draw.text((max(4, roi[0]), max(4, roi[1] - 20)), class_name, fill=(255, 255, 0))

        stem = sample["stem"]
        filename = f"{exported + 1:04d}_{stem}_{class_name}.jpg"
        preview.save(os.path.join(output_dir, filename), quality=95)
        exported += 1

    return exported


# ============================================================
# ОБУЧЕНИЕ ДЕТЕКТОРА
# ============================================================

def train_detection_model(
    train_samples: List[Dict],
    num_epochs: int = NUM_EPOCHS_DETECTOR,
    batch_size: int = BATCH_SIZE_DETECTOR,
    lr: float = LEARNING_RATE_DETECTOR,
    weight_decay: float = WEIGHT_DECAY_DETECTOR,
):
    device = get_device()
    print(f"[Detection] Device: {device}")

    subtrain_samples, val_samples = split_samples_for_detection(train_samples, val_ratio=0.2, seed=SEED)

    train_dataset = DetectionDataset(subtrain_samples, transform=DetectionTrainTransform())
    val_dataset = DetectionDataset(val_samples, transform=DetectionEvalTransform())

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=detection_collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=detection_collate_fn
    )

    model = build_detection_model(num_classes=2, pretrained=True).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=lr,
        momentum=0.9,
        weight_decay=weight_decay
    )

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.3)

    best_val_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0

        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad(set_to_none=True)
            losses.backward()
            optimizer.step()

            running_loss += float(losses.item())

        avg_train_loss = running_loss / max(len(train_loader), 1)
        val_loss = evaluate_detection_loss(model, val_loader, device)
        scheduler.step()

        print(
            f"[Detection] Epoch {epoch:02d}/{num_epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f}"
        )

        save_checkpoint(
            DETECTOR_LAST_PATH,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_metric=best_val_loss,
            extra={"task": "detection", "pipeline_version": PIPELINE_VERSION}
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                DETECTOR_BEST_PATH,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_metric=best_val_loss,
                extra={"task": "detection", "pipeline_version": PIPELINE_VERSION}
            )
            print(f"[Detection] Сохранены лучшие веса: {DETECTOR_BEST_PATH}")

    print("[Detection] Обучение завершено.")


# ============================================================
# ОБУЧЕНИЕ КЛАССИФИКАТОРА
# ============================================================

def train_kfold_classification_model(
    train_samples: List[Dict],
    demo_samples: List[Dict],
    fallback_roi_norm: Tuple[float, float, float, float],
    reference_size: Tuple[int, int],
    dataset_path: str,
    dataset_signature: str,
    num_epochs: int = NUM_EPOCHS_CLASSIFIER,
    batch_size: int = BATCH_SIZE_CLASSIFIER,
    learning_rate: float = LEARNING_RATE_CLASSIFIER,
    weight_decay: float = WEIGHT_DECAY_CLASSIFIER,
    n_splits: int = N_SPLITS,
    classifier_arch: str = CLASSIFIER_ARCH,
):
    set_seed(SEED)
    device = get_device()

    print(f"[Classification] Device: {device}")
    print(f"[Classification] Backbone: {classifier_arch}")

    if len(train_samples) == 0:
        raise RuntimeError("Пустой train-набор")
    if len(demo_samples) == 0:
        raise RuntimeError("Пустой demo-набор")

    train_labels_all = [sample["class_label"] for sample in train_samples]
    demo_labels_all = [sample["class_label"] for sample in demo_samples]

    print(f"[Classification] Train images: {len(train_samples)}")
    print(f"[Classification] Demo images (not used in training): {len(demo_samples)}")
    print(f"[Classification] Train classes: закрыта={train_labels_all.count(0)}, открыта={train_labels_all.count(1)}")
    print(f"[Classification] Demo classes: закрыта={demo_labels_all.count(0)}, открыта={demo_labels_all.count(1)}")
    print(f"[Classification] Fallback ROI (normalized): {fallback_roi_norm}")
    print(f"[Classification] Reference image size: {reference_size}")

    min_class_count = min(train_labels_all.count(0), train_labels_all.count(1))
    n_splits = min(n_splits, min_class_count)

    if n_splits < 2:
        raise RuntimeError("Слишком мало train-данных для k-fold")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    indices = np.arange(len(train_samples))

    all_fold_metrics = []
    best_thresholds = []
    best_overall_metric = -1.0
    best_overall_path = None

    for fold_idx, (subtrain_idx, val_idx) in enumerate(skf.split(indices, train_labels_all), start=1):
        print("=" * 90)
        print(f"[Classification] Fold {fold_idx}/{n_splits}")

        subtrain_samples = [train_samples[i] for i in subtrain_idx]
        val_samples = [train_samples[i] for i in val_idx]

        subtrain_dataset = ClassificationBoxCropDataset(
            samples=subtrain_samples,
            transform=get_classification_train_transform(),
            pad_ratio=DETECTION_PAD_RATIO
        )

        val_dataset = ClassificationBoxCropDataset(
            samples=val_samples,
            transform=get_classification_eval_transform(),
            pad_ratio=DETECTION_PAD_RATIO
        )

        subtrain_loader = DataLoader(
            subtrain_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True
        )

        model = build_classification_model(
            num_classes=2,
            pretrained=True,
            arch=classifier_arch
        ).to(device)
        trainable_params = [p for p in model.parameters() if p.requires_grad]

        subtrain_labels = [sample["class_label"] for sample in subtrain_samples]
        unique_classes = np.unique(subtrain_labels)

        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=unique_classes,
            y=subtrain_labels
        )

        weights_full = np.ones(2, dtype=np.float32)
        for cls, weight in zip(unique_classes, class_weights):
            weights_full[int(cls)] = float(weight)

        class_weights_tensor = torch.tensor(weights_full, dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=learning_rate,
            weight_decay=weight_decay
        )

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=learning_rate,
            epochs=num_epochs,
            steps_per_epoch=max(1, len(subtrain_loader)),
            pct_start=0.2,
            anneal_strategy="cos",
            div_factor=8.0,
            final_div_factor=80.0
        )

        scaler = torch.amp.GradScaler("cuda", enabled=True)

        best_fold_metric = -1.0
        best_fold_epoch = 0
        best_fold_threshold = DEFAULT_OPEN_THRESHOLD
        best_fold_cost = float("inf")
        patience_counter = 0
        fold_best_path = os.path.join(MODEL_DIR, f"classifier_fold_{fold_idx}_best.pth")

        for epoch in range(1, num_epochs + 1):
            model.train()
            running_loss = 0.0

            for images, targets in subtrain_loader:
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast("cuda", enabled=True):
                    logits = model(images)
                    loss = criterion(logits, targets)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                running_loss += float(loss.item())

            train_loss = running_loss / max(len(subtrain_loader), 1)

            val_loss, val_targets, val_prob_open = collect_probabilities(
                model=model,
                data_loader=val_loader,
                criterion=criterion,
                device=device
            )

            tuned_threshold, val_acc, val_bal_acc, val_f1, val_cost = find_best_open_threshold_cost_sensitive(
                targets=val_targets,
                prob_open=val_prob_open
            )

            print(
                f"[Classification][Fold {fold_idx}] Epoch {epoch:02d}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc * 100:.2f}% | "
                f"Val Bal Acc: {val_bal_acc * 100:.2f}% | "
                f"Val F1: {val_f1:.4f} | "
                f"Thr(open): {tuned_threshold:.3f} | "
                f"Cost: {val_cost:.1f} | "
                f"LR: {optimizer.param_groups[0]['lr']:.6f}"
            )

            better = False
            if val_bal_acc > best_fold_metric:
                better = True
            elif val_bal_acc == best_fold_metric and val_cost < best_fold_cost:
                better = True

            if better:
                best_fold_metric = val_bal_acc
                best_fold_epoch = epoch
                best_fold_threshold = tuned_threshold
                best_fold_cost = val_cost
                patience_counter = 0

                save_checkpoint(
                    save_path=fold_best_path,
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    best_metric=best_fold_metric,
                    extra={
                        "fold": fold_idx,
                        "model_type": get_model_type_name(),
                        "classifier_arch": classifier_arch,
                        "pipeline_version": PIPELINE_VERSION,
                        "open_threshold": best_fold_threshold,
                    },
                )
            else:
                patience_counter += 1

            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"[Classification][Fold {fold_idx}] Early stopping на эпохе {epoch}")
                break

        best_thresholds.append(best_fold_threshold)

        all_fold_metrics.append({
            "fold": fold_idx,
            "best_balanced_accuracy": float(best_fold_metric),
            "best_epoch": int(best_fold_epoch),
            "open_threshold": float(best_fold_threshold),
            "checkpoint_path": fold_best_path,
        })

        if best_fold_metric > best_overall_metric:
            best_overall_metric = best_fold_metric
            best_overall_path = fold_best_path

    final_open_threshold = float(np.median(best_thresholds)) if best_thresholds else DEFAULT_OPEN_THRESHOLD

    save_meta(
        meta_path=META_PATH,
        fallback_roi_norm=fallback_roi_norm,
        reference_size=reference_size,
        open_threshold=final_open_threshold,
        train_count=len(train_samples),
        demo_count=len(demo_samples),
        dataset_path=os.path.abspath(dataset_path),
        dataset_signature=dataset_signature,
    )

    summary = {
        "pipeline_version": PIPELINE_VERSION,
        "model_type": get_model_type_name(),
        "classifier_arch": classifier_arch,
        "dataset_path": os.path.abspath(dataset_path),
        "dataset_signature": dataset_signature,
        "fallback_roi_norm": list(fallback_roi_norm),
        "reference_size": list(reference_size),
        "open_threshold": final_open_threshold,
        "uncertainty_margin": UNCERTAINTY_MARGIN,
        "false_open_cost": FALSE_OPEN_COST,
        "false_closed_cost": FALSE_CLOSED_COST,
        "train_count": len(train_samples),
        "demo_count": len(demo_samples),
        "folds": all_fold_metrics,
        "detector_best_path": DETECTOR_BEST_PATH,
        "best_overall_classifier_checkpoint": best_overall_path,
        "mean_balanced_accuracy": float(np.mean([m["best_balanced_accuracy"] for m in all_fold_metrics])),
    }

    with open(TRAINING_SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("=" * 90)
    print("[Classification] Обучение завершено")
    print(f"[Classification] Лучшая модель: {best_overall_path}")
    print(f"[Classification] Финальный tuned threshold (open): {final_open_threshold:.3f}")
    print(f"[Classification] Средняя balanced accuracy по фолдам: {summary['mean_balanced_accuracy'] * 100:.2f}%")


# ============================================================
# ЗАГРУЗКА МОДЕЛЕЙ
# ============================================================

def load_detector_model(model_path: str, device: torch.device):
    checkpoint = torch.load(model_path, map_location=device)
    model = build_detection_model(num_classes=2, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def load_classifier_fold_models(
    model_dir: str,
    device: torch.device,
    default_arch: str = CLASSIFIER_ARCH
):
    checkpoint_paths = get_classifier_fold_checkpoint_paths(model_dir)

    if not checkpoint_paths:
        raise FileNotFoundError("Не найдено ни одного файла classifier_fold_*_best.pth")

    models_list = []
    thresholds = []

    for path in checkpoint_paths:
        checkpoint = torch.load(path, map_location=device)

        arch = str(checkpoint.get("classifier_arch", default_arch)).lower()
        model = build_classification_model(
            num_classes=2,
            pretrained=False,
            arch=arch
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()

        models_list.append(model)
        thresholds.append(float(checkpoint.get("open_threshold", DEFAULT_OPEN_THRESHOLD)))

    return models_list, thresholds


# ============================================================
# PREDICTOR
# ============================================================

class CashRegisterPredictor:
    def __init__(
        self,
        detector_path: str = DETECTOR_BEST_PATH,
        model_dir: str = MODEL_DIR,
        meta_path: str = META_PATH,
        device: Optional[torch.device] = None
    ):
        self.device = device if device is not None else get_device()
        self.meta = load_meta(meta_path)
        self.classifier_arch = str(self.meta.get("classifier_arch", CLASSIFIER_ARCH)).lower()
        self.model_type = f"FasterRCNN + {self.classifier_arch}"

        self.detector = load_detector_model(detector_path, self.device)
        self.classifier_models, self.fold_thresholds = load_classifier_fold_models(
            model_dir=model_dir,
            device=self.device,
            default_arch=self.classifier_arch
        )

        self.open_threshold = float(self.meta.get("open_threshold", DEFAULT_OPEN_THRESHOLD))
        self.uncertainty_margin = float(self.meta.get("uncertainty_margin", UNCERTAINTY_MARGIN))
        self.fallback_roi_norm = tuple(self.meta["fallback_roi_norm"])

        self.eval_transform = get_classification_eval_transform()

    def detect_cash_register(
        self,
        image: Image.Image
    ) -> Tuple[Tuple[int, int, int, int], float, str]:
        image_w, image_h = image.size
        image_tensor = TF.to_tensor(image).to(self.device)

        with torch.no_grad():
            output = self.detector([image_tensor])[0]

        boxes = output["boxes"].detach().cpu().numpy()
        scores = output["scores"].detach().cpu().numpy()

        if len(boxes) > 0:
            keep = scores >= DETECTION_SCORE_THRESHOLD
            boxes = boxes[keep]
            scores = scores[keep]

            if len(boxes) > 0:
                best_idx = int(np.argmax(scores))
                best_box = tuple(map(float, boxes[best_idx].tolist()))
                best_score = float(scores[best_idx])

                expanded = expand_box(best_box, image_w, image_h, pad_ratio=DETECTION_PAD_RATIO)
                return expanded, best_score, "detector"

        fallback_box = normalized_roi_to_pixels(self.fallback_roi_norm, image_w, image_h)
        return fallback_box, 0.0, "fallback_roi"

    def classify_from_box(
        self,
        image: Image.Image,
        box: Tuple[int, int, int, int]
    ) -> Tuple[np.ndarray, Image.Image]:
        image_w, image_h = image.size
        candidate_boxes = make_inference_candidate_boxes(box, image_w, image_h)

        probs_sum = np.zeros(2, dtype=np.float32)
        total_weight = 0.0

        preview_crop = crop_by_box(image, candidate_boxes[0][0])

        with torch.no_grad():
            for candidate_box, weight in candidate_boxes:
                crop = crop_by_box(image, candidate_box)
                tensor = self.eval_transform(crop).unsqueeze(0).to(self.device)

                probs_models = []
                for model in self.classifier_models:
                    logits = model(tensor)
                    probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
                    probs_models.append(probs)

                mean_probs_candidate = np.mean(np.stack(probs_models, axis=0), axis=0)
                probs_sum += weight * mean_probs_candidate
                total_weight += weight

        final_probs = probs_sum / max(total_weight, 1e-8)
        return final_probs, preview_crop

    def decide_state(self, prob_open: float, prob_closed: float) -> Tuple[str, int, float]:
        threshold = self.open_threshold

        if prob_open >= threshold + self.uncertainty_margin:
            return CLASS_NAMES[1], 1, prob_open

        if prob_open <= threshold - self.uncertainty_margin:
            return CLASS_NAMES[0], 0, prob_closed

        return "состояние не определено", int(prob_open >= threshold), max(prob_open, prob_closed)

    def predict(self, image_path: str) -> Dict:
        image = Image.open(image_path).convert("RGB")

        box, detection_score, source = self.detect_cash_register(image)
        probs, crop_preview = self.classify_from_box(image, box)

        prob_closed = float(probs[0])
        prob_open = float(probs[1])

        state, predicted_class, confidence = self.decide_state(prob_open, prob_closed)

        return {
            "state": state,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "probabilities": {
                "закрыта": prob_closed,
                "открыта": prob_open,
            },
            "open_threshold": self.open_threshold,
            "detection_score": detection_score,
            "roi_source": source,
            "roi_box": box,
            "image": image,
            "crop": crop_preview,
            "model_type": self.model_type,
        }


# ============================================================
# GUI
# ============================================================

def draw_roi_on_image(image: Image.Image, roi_box: Tuple[int, int, int, int]) -> Image.Image:
    preview = image.copy()
    draw = ImageDraw.Draw(preview)
    draw.rectangle(roi_box, outline=(255, 0, 0), width=4)
    return preview


def fit_image_for_gui(image: Image.Image, max_size: Tuple[int, int]) -> Image.Image:
    preview = image.copy()
    preview.thumbnail(max_size, Image.LANCZOS)
    return preview


def extract_source_stem_from_demo_filename(file_path: str) -> Optional[str]:
    name = os.path.splitext(os.path.basename(file_path))[0]
    marker = "_class_"
    if marker not in name:
        return None
    return name.split(marker, 1)[0]


class CashRegisterGUI:
    def __init__(self, predictor: CashRegisterPredictor, dataset_path: str = DEFAULT_DATASET_PATH):
        self.predictor = predictor
        self.dataset_path = dataset_path
        self.train_stems = set()
        self._load_demo_meta()

        self.root = tk.Tk()
        self.root.title("Классификация состояния кассового аппарата")
        self.root.geometry("1500x940")

        self.current_image_path = None
        self.tk_original = None
        self.tk_crop = None

        self.build_ui()
        self.refresh_demo_count()

    def _load_demo_meta(self):
        if not os.path.exists(DEMO_META_PATH):
            self.train_stems = set()
            return
        try:
            with open(DEMO_META_PATH, "r", encoding="utf-8") as f:
                meta = json.load(f)
            self.train_stems = set(meta.get("train_stems", []))
        except Exception:
            self.train_stems = set()

    def build_ui(self):
        top_frame = ttk.Frame(self.root, padding=10)
        top_frame.pack(fill="x")

        ttk.Button(top_frame, text="Выбрать demo-изображение", command=self.choose_image).pack(side="left", padx=5)
        ttk.Button(top_frame, text="Классифицировать", command=self.classify_current_image).pack(side="left", padx=5)
        ttk.Button(top_frame, text="Пересоздать demo-аугментации", command=self.regenerate_demo_images).pack(side="left", padx=5)
        ttk.Button(top_frame, text="Открыть папку demo_images", command=self.open_demo_folder).pack(side="left", padx=5)
        ttk.Button(top_frame, text="Выход", command=self.root.destroy).pack(side="left", padx=5)

        self.path_label = ttk.Label(top_frame, text="Файл не выбран")
        self.path_label.pack(side="left", padx=15)

        info_frame = ttk.Frame(self.root, padding=(10, 0, 10, 10))
        info_frame.pack(fill="x")

        info_text = (
            "Интерфейс работает только с отложенными demo-изображениями. "
            "Они не участвовали в обучении. Аугментированные demo-картинки "
            "нужны только для демонстрации и увеличения выбора в интерфейсе."
        )
        self.info_label = ttk.Label(
            info_frame,
            text=info_text,
            foreground="darkred",
            font=("Arial", 10, "bold")
        )
        self.info_label.pack(anchor="w")

        self.demo_count_label = ttk.Label(
            info_frame,
            text="Доступно demo-изображений: -",
            font=("Arial", 10, "bold"),
            foreground="darkgreen"
        )
        self.demo_count_label.pack(anchor="w", pady=(5, 0))

        center_frame = ttk.Frame(self.root, padding=10)
        center_frame.pack(fill="both", expand=True)

        left_panel = ttk.LabelFrame(center_frame, text="Исходное изображение", padding=10)
        left_panel.pack(side="left", fill="both", expand=True, padx=5, pady=5)

        right_panel = ttk.LabelFrame(center_frame, text="Область интереса (ROI)", padding=10)
        right_panel.pack(side="right", fill="y", padx=5, pady=5)

        self.original_image_label = ttk.Label(left_panel)
        self.original_image_label.pack(fill="both", expand=True)

        self.crop_image_label = ttk.Label(right_panel)
        self.crop_image_label.pack(pady=10)

        self.model_label = ttk.Label(
            right_panel,
            text=f"Модель: {get_model_type_name()}",
            font=("Arial", 12, "bold")
        )
        self.model_label.pack(anchor="w", pady=5)

        self.result_label = ttk.Label(
            right_panel,
            text="Результат: -",
            font=("Arial", 16, "bold"),
            foreground="blue",
        )
        self.result_label.pack(anchor="w", pady=(20, 10))

        self.conf_label = ttk.Label(right_panel, text="Уверенность: -", font=("Arial", 12))
        self.conf_label.pack(anchor="w", pady=5)

        self.threshold_label = ttk.Label(right_panel, text="Threshold(open): -", font=("Arial", 12))
        self.threshold_label.pack(anchor="w", pady=5)

        self.det_score_label = ttk.Label(right_panel, text="Detection score: -", font=("Arial", 12))
        self.det_score_label.pack(anchor="w", pady=5)

        self.roi_source_label = ttk.Label(right_panel, text="Источник ROI: -", font=("Arial", 12))
        self.roi_source_label.pack(anchor="w", pady=5)

        self.prob_closed_label = ttk.Label(right_panel, text="P(закрыта): -", font=("Arial", 12))
        self.prob_closed_label.pack(anchor="w", pady=5)

        self.prob_open_label = ttk.Label(right_panel, text="P(открыта): -", font=("Arial", 12))
        self.prob_open_label.pack(anchor="w", pady=5)

        self.roi_label = ttk.Label(right_panel, text="ROI: -", font=("Arial", 11))
        self.roi_label.pack(anchor="w", pady=5)

    def refresh_demo_count(self):
        count = count_image_files_in_dir(DEMO_POOL_DIR)
        self.demo_count_label.config(text=f"Доступно demo-изображений: {count}")

    def open_demo_folder(self):
        ensure_dir(DEMO_POOL_DIR)
        try:
            os.startfile(os.path.abspath(DEMO_POOL_DIR))
        except Exception as exc:
            messagebox.showerror("Ошибка", f"Не удалось открыть папку:\n{exc}")

    def regenerate_demo_images(self):
        try:
            regenerate_demo_pool_from_saved_split(
                dataset_path=self.dataset_path,
                split_info_path=SPLIT_INFO_PATH,
                demo_pool_dir=DEMO_POOL_DIR,
                demo_meta_path=DEMO_META_PATH,
                copies_per_image=DEMO_AUG_COPIES_PER_IMAGE,
                seed=None
            )
            self._load_demo_meta()
            self.refresh_demo_count()
            messagebox.showinfo(
                "Готово",
                "Demo-аугментации пересозданы.\n"
                "Теперь в папке demo_images появился обновлённый набор изображений."
            )
        except Exception as exc:
            messagebox.showerror("Ошибка", str(exc))

    def choose_image(self):
        ensure_dir(DEMO_POOL_DIR)

        file_path = filedialog.askopenfilename(
            title="Выберите demo-изображение",
            initialdir=os.path.abspath(DEMO_POOL_DIR),
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.webp"),
                ("All files", "*.*"),
            ],
        )

        if not file_path:
            return

        if not is_path_inside_dir(file_path, DEMO_POOL_DIR):
            messagebox.showwarning(
                "Недопустимый файл",
                "Можно выбирать только изображения из папки demo_images."
            )
            return

        stem = extract_source_stem_from_demo_filename(file_path)
        if stem is not None and stem in self.train_stems:
            messagebox.showwarning(
                "Недопустимое demo-изображение",
                "Выбранное изображение пересекается с train-набором. "
                "Обновите demo pool."
            )
            return

        self.current_image_path = file_path
        self.path_label.config(text=file_path)

        image = Image.open(file_path).convert("RGB")
        self.show_original_image(image)

    def show_original_image(self, image: Image.Image, roi_box: Optional[Tuple[int, int, int, int]] = None):
        preview = image
        if roi_box is not None:
            preview = draw_roi_on_image(preview, roi_box)

        preview = fit_image_for_gui(preview, GUI_ORIGINAL_MAX_SIZE)
        self.tk_original = ImageTk.PhotoImage(preview)
        self.original_image_label.config(image=self.tk_original)

    def show_crop_image(self, crop: Image.Image):
        preview = fit_image_for_gui(crop, GUI_CROP_MAX_SIZE)
        self.tk_crop = ImageTk.PhotoImage(preview)
        self.crop_image_label.config(image=self.tk_crop)

    def classify_current_image(self):
        if not self.current_image_path:
            messagebox.showwarning("Предупреждение", "Сначала выберите demo-изображение")
            return

        if not is_path_inside_dir(self.current_image_path, DEMO_POOL_DIR):
            messagebox.showwarning(
                "Недопустимый файл",
                "Для демонстрации можно использовать только изображения из папки demo_images."
            )
            return

        try:
            result = self.predictor.predict(self.current_image_path)
        except Exception as exc:
            messagebox.showerror("Ошибка", str(exc))
            return

        self.show_original_image(result["image"], result["roi_box"])
        self.show_crop_image(result["crop"])

        self.model_label.config(text=f"Модель: {result['model_type']}")
        self.result_label.config(text=f"Результат: {result['state']}")
        self.conf_label.config(text=f"Уверенность: {result['confidence'] * 100:.2f}%")
        self.threshold_label.config(text=f"Threshold(open): {result['open_threshold']:.3f}")
        self.det_score_label.config(text=f"Detection score: {result['detection_score'] * 100:.2f}%")
        self.roi_source_label.config(text=f"Источник ROI: {result['roi_source']}")
        self.prob_closed_label.config(text=f"P(закрыта): {result['probabilities']['закрыта'] * 100:.2f}%")
        self.prob_open_label.config(text=f"P(открыта): {result['probabilities']['открыта'] * 100:.2f}%")
        self.roi_label.config(text=f"ROI: {result['roi_box']}")

    def run(self):
        self.root.mainloop()


# ============================================================
# PIPELINE ORCHESTRATION
# ============================================================

def prepare_dataset_context(dataset_path: str):
    all_samples = build_samples(dataset_path)
    class_counts = {
        0: sum(1 for sample in all_samples if int(sample["class_label"]) == 0),
        1: sum(1 for sample in all_samples if int(sample["class_label"]) == 1),
    }

    if class_counts[0] == 0 or class_counts[1] == 0:
        raise RuntimeError(
            "В датасете должен быть баланс двух классов: 0=закрыта и 1=открыта. "
            f"Сейчас: class0={class_counts[0]}, class1={class_counts[1]}."
        )

    if min(class_counts[0], class_counts[1]) < 2:
        raise RuntimeError(
            "Слишком мало примеров одного из классов. "
            "Для корректного split train/demo нужно минимум по 2 изображения каждого класса."
        )

    dataset_signature = compute_dataset_signature(all_samples)

    fallback_roi_norm, reference_size = compute_fallback_roi_from_labels(all_samples)
    train_samples, demo_samples, split_was_created = create_or_load_demo_split(
        samples=all_samples,
        split_info_path=SPLIT_INFO_PATH,
        demo_ratio=DEMO_RATIO
    )
    validate_train_demo_disjoint(train_samples=train_samples, demo_samples=demo_samples)

    return {
        "all_samples": all_samples,
        "class_counts": class_counts,
        "dataset_signature": dataset_signature,
        "fallback_roi_norm": fallback_roi_norm,
        "reference_size": reference_size,
        "train_samples": train_samples,
        "demo_samples": demo_samples,
        "split_was_created": split_was_created,
    }


def run_training_pipeline(
    dataset_path: str,
    force_train: bool = False,
    rebuild_demo: bool = False
):
    context = prepare_dataset_context(dataset_path)

    print(f"[Main] Dataset path: {dataset_path}")
    print(
        f"[Main] Samples: {len(context['all_samples'])} | "
        f"закрыта={context['class_counts'][0]} | "
        f"открыта={context['class_counts'][1]}"
    )
    print(
        f"[Main] Split: train={len(context['train_samples'])}, demo={len(context['demo_samples'])}"
    )

    demo_rebuilt = ensure_demo_pool(
        train_samples=context["train_samples"],
        demo_samples=context["demo_samples"],
        dataset_signature=context["dataset_signature"],
        force_rebuild=rebuild_demo or context["split_was_created"],
        demo_pool_dir=DEMO_POOL_DIR,
        demo_meta_path=DEMO_META_PATH,
        copies_per_image=DEMO_AUG_COPIES_PER_IMAGE,
        seed=SEED
    )
    if demo_rebuilt:
        print(f"[Main] Demo pool regenerated: {DEMO_POOL_DIR}")
    else:
        print(f"[Main] Demo pool is up to date: {DEMO_POOL_DIR}")

    ready = has_ready_models(
        model_dir=MODEL_DIR,
        meta_path=META_PATH,
        expected_dataset_signature=context["dataset_signature"]
    )
    if ready and not force_train:
        print("[Main] Compatible best weights already exist. Skipping training.")
        return context

    if force_train and ready:
        print("[Main] --force-train enabled. Retraining from scratch.")
    else:
        print("[Main] Starting training on GPU.")

    train_detection_model(train_samples=context["train_samples"])
    train_kfold_classification_model(
        train_samples=context["train_samples"],
        demo_samples=context["demo_samples"],
        fallback_roi_norm=context["fallback_roi_norm"],
        reference_size=context["reference_size"],
        dataset_path=dataset_path,
        dataset_signature=context["dataset_signature"],
        classifier_arch=CLASSIFIER_ARCH
    )

    return context


def run_gui_mode(
    dataset_path: str,
    rebuild_demo: bool = False
):
    context = prepare_dataset_context(dataset_path)
    ensure_demo_pool(
        train_samples=context["train_samples"],
        demo_samples=context["demo_samples"],
        dataset_signature=context["dataset_signature"],
        force_rebuild=rebuild_demo,
        demo_pool_dir=DEMO_POOL_DIR,
        demo_meta_path=DEMO_META_PATH,
        copies_per_image=DEMO_AUG_COPIES_PER_IMAGE,
        seed=SEED
    )

    if not has_ready_models(
        model_dir=MODEL_DIR,
        meta_path=META_PATH,
        expected_dataset_signature=context["dataset_signature"]
    ):
        raise RuntimeError(
            "No compatible trained weights found for current dataset. "
            "Run with --mode train first."
        )

    device = get_device()
    predictor = CashRegisterPredictor(
        detector_path=DETECTOR_BEST_PATH,
        model_dir=MODEL_DIR,
        meta_path=META_PATH,
        device=device
    )
    app = CashRegisterGUI(predictor, dataset_path=dataset_path)
    app.run()


def run_preview_dataset_mode(dataset_path: str, preview_count: int):
    samples = build_samples(dataset_path)
    preview_dir = os.path.join(OUTPUT_DIR, "dataset_preview")
    count = export_dataset_preview(
        samples=samples,
        output_dir=preview_dir,
        max_items=preview_count,
        seed=SEED
    )
    print(f"[Preview] Exported {count} preview images to: {os.path.abspath(preview_dir)}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    args = parse_args()
    dataset_path = os.path.abspath(args.dataset_path)
    mode = "train" if args.skip_gui else args.mode

    set_seed(SEED)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    ensure_dir(OUTPUT_DIR)
    ensure_dir(MODEL_DIR)

    if mode == "train":
        run_training_pipeline(
            dataset_path=dataset_path,
            force_train=args.force_train,
            rebuild_demo=args.rebuild_demo
        )
    elif mode == "gui":
        run_gui_mode(
            dataset_path=dataset_path,
            rebuild_demo=args.rebuild_demo
        )
    elif mode == "train_and_gui":
        run_training_pipeline(
            dataset_path=dataset_path,
            force_train=args.force_train,
            rebuild_demo=args.rebuild_demo
        )
        run_gui_mode(
            dataset_path=dataset_path,
            rebuild_demo=False
        )
    elif mode == "preview_dataset":
        run_preview_dataset_mode(
            dataset_path=dataset_path,
            preview_count=args.preview_count
        )
    else:
        raise RuntimeError(f"Unsupported mode: {mode}")

