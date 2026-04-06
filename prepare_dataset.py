import argparse
import ast
import hashlib
import json
import os
import re
import shutil
import sys
import urllib.request
import zipfile
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
DEFAULT_OPEN_HINTS = ["open", "opened", "draweropen", "cashregisteropen", "otkryta"]
DEFAULT_CLOSE_HINTS = ["close", "closed", "drawerclose", "cashregisterclose", "zakryta"]


@dataclass
class SampleEntry:
    image_path: str
    class_label: int
    box_xyxy: Tuple[float, float, float, float]
    source: str
    source_stem: str


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def clear_dir(path: str) -> None:
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def log_progress(message: str) -> None:
    print(f"[prepare_dataset] {message}", file=sys.stderr, flush=True)


def download_file(url: str, dst_path: str, timeout_sec: int = 180, retries: int = 2) -> None:
    tmp_path = dst_path + ".part"
    last_exc = None
    for _ in range(max(0, int(retries)) + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "python-urllib"})
            with urllib.request.urlopen(request, timeout=timeout_sec) as response, open(tmp_path, "wb") as out_f:
                shutil.copyfileobj(response, out_f)
            os.replace(tmp_path, dst_path)
            return
        except Exception as exc:
            last_exc = exc
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
    raise RuntimeError(f"Failed to download {url} -> {dst_path}: {last_exc}")


def is_image_file(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in IMAGE_EXTENSIONS


def safe_read_text(path: str) -> str:
    for encoding in ("utf-8", "utf-8-sig", "cp1251", "latin-1"):
        try:
            with open(path, "r", encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def normalize_token(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "", text.lower())


def normalize_names_field(raw_names) -> Dict[int, str]:
    names: Dict[int, str] = {}

    if isinstance(raw_names, list):
        for idx, value in enumerate(raw_names):
            names[idx] = str(value).strip()
        return names

    if isinstance(raw_names, dict):
        for key, value in raw_names.items():
            try:
                idx = int(key)
            except (ValueError, TypeError):
                continue
            names[idx] = str(value).strip()
        return names

    return names


def parse_names_fallback(yaml_text: str) -> Dict[int, str]:
    lines = yaml_text.splitlines()
    names: Dict[int, str] = {}

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped.startswith("names:"):
            continue

        inline_value = stripped.split(":", 1)[1].strip()
        if inline_value:
            try:
                parsed = ast.literal_eval(inline_value)
                names = normalize_names_field(parsed)
                if names:
                    return names
            except Exception:
                pass

            if inline_value.startswith("[") and inline_value.endswith("]"):
                raw_items = inline_value[1:-1].split(",")
                cleaned = [item.strip().strip("'\"") for item in raw_items if item.strip()]
                names = {idx: item for idx, item in enumerate(cleaned)}
                if names:
                    return names

        base_indent = len(line) - len(line.lstrip(" "))
        j = i + 1
        while j < len(lines):
            block_line = lines[j]
            block_stripped = block_line.strip()

            if not block_stripped or block_stripped.startswith("#"):
                j += 1
                continue

            indent = len(block_line) - len(block_line.lstrip(" "))
            if indent <= base_indent:
                break

            match = re.match(r"^\s*(\d+)\s*:\s*(.+?)\s*$", block_line)
            if match:
                class_id = int(match.group(1))
                value = match.group(2).strip().strip("'\"")
                names[class_id] = value
            j += 1

        if names:
            return names

    return names


def parse_data_yaml_names(data_yaml_path: str) -> Dict[int, str]:
    yaml_text = safe_read_text(data_yaml_path)

    # Prefer PyYAML if available, then fallback to lightweight parsing.
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(yaml_text) or {}
        names = normalize_names_field(data.get("names"))
        if names:
            return names
    except Exception:
        pass

    return parse_names_fallback(yaml_text)


def infer_state_map(
    names: Dict[int, str],
    close_hints: Iterable[str],
    open_hints: Iterable[str],
    close_class_ids: Optional[List[int]] = None,
    open_class_ids: Optional[List[int]] = None,
) -> Dict[int, int]:
    state_map: Dict[int, int] = {}

    if close_class_ids:
        for class_id in close_class_ids:
            state_map[int(class_id)] = 0

    if open_class_ids:
        for class_id in open_class_ids:
            state_map[int(class_id)] = 1

    close_hint_tokens = [normalize_token(hint) for hint in close_hints]
    open_hint_tokens = [normalize_token(hint) for hint in open_hints]

    for class_id, class_name in names.items():
        token = normalize_token(class_name)
        if any(hint and hint in token for hint in close_hint_tokens):
            state_map[class_id] = 0
        if any(hint and hint in token for hint in open_hint_tokens):
            state_map[class_id] = 1

    # Fallback: binary dataset with two classes and no explicit mapping.
    if not state_map and len(names) == 2:
        sorted_ids = sorted(names.keys())
        state_map[sorted_ids[0]] = 0
        state_map[sorted_ids[1]] = 1

    # Keep only mapped classes that exist in names when names are known.
    if names:
        state_map = {cid: state for cid, state in state_map.items() if cid in names}

    return state_map


def build_global_image_index(root_dir: str) -> Dict[str, List[str]]:
    image_index: Dict[str, List[str]] = {}
    for current_root, _, files in os.walk(root_dir):
        for filename in files:
            _, ext = os.path.splitext(filename)
            if ext.lower() not in IMAGE_EXTENSIONS:
                continue
            stem, _ = os.path.splitext(filename)
            image_path = os.path.join(current_root, filename)
            image_index.setdefault(stem, []).append(image_path)
    return image_index


def find_image_for_label(label_path: str, image_index: Dict[str, List[str]]) -> Optional[str]:
    label_dir = os.path.dirname(label_path)
    stem = os.path.splitext(os.path.basename(label_path))[0]

    candidate_dirs = []

    parts = label_dir.split(os.sep)
    for i, part in enumerate(parts):
        if part.lower() == "labels":
            candidate_dirs.append(os.sep.join(parts[:i] + ["images"] + parts[i + 1:]))

    candidate_dirs.append(label_dir)

    for candidate_dir in candidate_dirs:
        for ext in IMAGE_EXTENSIONS:
            candidate_path = os.path.join(candidate_dir, stem + ext)
            if os.path.isfile(candidate_path):
                return candidate_path

    stem_matches = image_index.get(stem, [])
    if len(stem_matches) == 1:
        return stem_matches[0]
    if len(stem_matches) > 1:
        # Prefer paths with the most similar folder structure.
        label_parts = label_path.lower().split(os.sep)
        best_path = None
        best_score = -1
        for image_path in stem_matches:
            image_parts = image_path.lower().split(os.sep)
            score = len(set(label_parts) & set(image_parts))
            if score > best_score:
                best_score = score
                best_path = image_path
        return best_path

    return None


def parse_internal_box_file(label_path: str) -> Optional[Tuple[float, float, float, float]]:
    best_box = None
    best_area = -1.0

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 4:
                continue
            try:
                x1, y1, x2, y2 = map(float, parts)
            except ValueError:
                continue
            if x2 < x1:
                x1, x2 = x2, x1
            if y2 < y1:
                y1, y2 = y2, y1
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            if area > best_area:
                best_area = area
                best_box = (x1, y1, x2, y2)

    return best_box


def collect_internal_entries(dataset_root: str, source_tag: str) -> List[SampleEntry]:
    images_dir = os.path.join(dataset_root, "images")
    det_dir = os.path.join(dataset_root, "labels_detection")
    cls_dir = os.path.join(dataset_root, "labels_classification")

    if not (os.path.isdir(images_dir) and os.path.isdir(det_dir) and os.path.isdir(cls_dir)):
        return []

    image_by_stem = {}
    for filename in os.listdir(images_dir):
        path = os.path.join(images_dir, filename)
        if os.path.isfile(path) and is_image_file(path):
            stem, _ = os.path.splitext(filename)
            image_by_stem[stem] = path

    det_by_stem = {}
    for filename in os.listdir(det_dir):
        path = os.path.join(det_dir, filename)
        stem, ext = os.path.splitext(filename)
        if os.path.isfile(path) and ext.lower() == ".txt":
            det_by_stem[stem] = path

    cls_by_stem = {}
    for filename in os.listdir(cls_dir):
        path = os.path.join(cls_dir, filename)
        stem, ext = os.path.splitext(filename)
        if os.path.isfile(path) and ext.lower() == ".txt":
            cls_by_stem[stem] = path

    common_stems = sorted(set(image_by_stem.keys()) & set(det_by_stem.keys()) & set(cls_by_stem.keys()))

    result: List[SampleEntry] = []
    for stem in common_stems:
        try:
            with open(cls_by_stem[stem], "r", encoding="utf-8") as f:
                class_label = int(f.readline().strip())
        except Exception:
            continue

        if class_label not in (0, 1):
            continue

        box = parse_internal_box_file(det_by_stem[stem])
        if box is None:
            continue

        result.append(
            SampleEntry(
                image_path=image_by_stem[stem],
                class_label=class_label,
                box_xyxy=box,
                source=source_tag,
                source_stem=stem,
            )
        )

    return result


def yolo_to_xyxy(
    xc: float,
    yc: float,
    bw: float,
    bh: float,
    image_w: int,
    image_h: int,
) -> Optional[Tuple[float, float, float, float]]:
    x1 = (xc - bw / 2.0) * image_w
    y1 = (yc - bh / 2.0) * image_h
    x2 = (xc + bw / 2.0) * image_w
    y2 = (yc + bh / 2.0) * image_h

    x1 = max(0.0, min(float(image_w - 1), x1))
    y1 = max(0.0, min(float(image_h - 1), y1))
    x2 = max(x1 + 1.0, min(float(image_w), x2))
    y2 = max(y1 + 1.0, min(float(image_h), y2))

    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def collect_yolo_entries(yolo_root: str, state_map: Dict[int, int], source_tag: str) -> Tuple[List[SampleEntry], Dict]:
    label_paths: List[str] = []
    for current_root, _, files in os.walk(yolo_root):
        for filename in files:
            if filename.lower().endswith(".txt"):
                label_paths.append(os.path.join(current_root, filename))
    label_paths.sort()

    image_index = build_global_image_index(yolo_root)

    stats = {
        "labels_total": len(label_paths),
        "missing_image": 0,
        "no_target_classes": 0,
        "invalid_labels": 0,
        "kept": 0,
        "class_counts": {"0": 0, "1": 0},
    }

    entries: List[SampleEntry] = []

    for label_path in label_paths:
        image_path = find_image_for_label(label_path, image_index)
        if image_path is None:
            stats["missing_image"] += 1
            continue

        try:
            with Image.open(image_path) as image:
                image_w, image_h = image.size
        except Exception:
            stats["missing_image"] += 1
            continue

        best_candidate = None
        best_area = -1.0
        valid_lines_found = False

        with open(label_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                try:
                    class_id = int(float(parts[0]))
                    xc = float(parts[1])
                    yc = float(parts[2])
                    bw = float(parts[3])
                    bh = float(parts[4])
                except ValueError:
                    stats["invalid_labels"] += 1
                    continue

                valid_lines_found = True

                if class_id not in state_map:
                    continue

                box = yolo_to_xyxy(xc, yc, bw, bh, image_w, image_h)
                if box is None:
                    continue

                x1, y1, x2, y2 = box
                area = (x2 - x1) * (y2 - y1)
                if area > best_area:
                    best_area = area
                    best_candidate = (box, state_map[class_id])

        if best_candidate is None:
            if valid_lines_found:
                stats["no_target_classes"] += 1
            else:
                stats["invalid_labels"] += 1
            continue

        box, class_label = best_candidate
        stem = os.path.splitext(os.path.basename(image_path))[0]
        entries.append(
            SampleEntry(
                image_path=image_path,
                class_label=int(class_label),
                box_xyxy=box,
                source=source_tag,
                source_stem=stem,
            )
        )
        stats["kept"] += 1
        stats["class_counts"][str(class_label)] += 1

    return entries, stats


def file_sha1(path: str) -> str:
    digest = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def write_internal_dataset(
    entries: List[SampleEntry],
    target_root: str,
    replace_target: bool = False,
    dedupe: bool = True,
) -> Dict:
    log_progress(f"Writing internal dataset to: {target_root}")
    if replace_target:
        clear_dir(target_root)
    else:
        ensure_dir(target_root)

    images_dir = os.path.join(target_root, "images")
    det_dir = os.path.join(target_root, "labels_detection")
    cls_dir = os.path.join(target_root, "labels_classification")

    clear_dir(images_dir)
    clear_dir(det_dir)
    clear_dir(cls_dir)

    seen_hashes = set()
    skipped_duplicates = 0
    written = 0
    class_counts = {0: 0, 1: 0}
    source_counts: Dict[str, int] = {}
    total_entries = len(entries)
    if total_entries == 0:
        log_progress("No entries to write.")

    for index, entry in enumerate(entries, start=1):
        if index == 1 or index % 200 == 0 or index == total_entries:
            log_progress(f"Writing sample {index}/{total_entries}")
        if dedupe:
            image_hash = file_sha1(entry.image_path)
            dedupe_key = f"{image_hash}:{entry.class_label}"
            if dedupe_key in seen_hashes:
                skipped_duplicates += 1
                continue
            seen_hashes.add(dedupe_key)

        written += 1
        stem = f"{written:06d}"
        ext = os.path.splitext(entry.image_path)[1].lower()
        if ext not in IMAGE_EXTENSIONS:
            ext = ".jpg"

        image_dst = os.path.join(images_dir, stem + ext)
        det_dst = os.path.join(det_dir, stem + ".txt")
        cls_dst = os.path.join(cls_dir, stem + ".txt")

        shutil.copy2(entry.image_path, image_dst)

        x1, y1, x2, y2 = entry.box_xyxy
        with open(det_dst, "w", encoding="utf-8") as f:
            f.write(f"{x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f}\n")

        with open(cls_dst, "w", encoding="utf-8") as f:
            f.write(f"{int(entry.class_label)}\n")

        class_counts[int(entry.class_label)] += 1
        source_counts[entry.source] = source_counts.get(entry.source, 0) + 1

    summary = {
        "output_samples": written,
        "output_class_counts": {"0": class_counts[0], "1": class_counts[1]},
        "output_source_counts": source_counts,
        "skipped_duplicates": skipped_duplicates,
    }

    summary_path = os.path.join(target_root, "prepare_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary


def find_first_data_yaml(root_dir: str) -> Tuple[str, str]:
    candidates = []
    for current_root, _, files in os.walk(root_dir):
        for filename in files:
            if filename.lower() == "data.yaml":
                candidates.append(os.path.join(current_root, filename))
    if not candidates:
        raise FileNotFoundError(f"Could not find data.yaml under: {root_dir}")
    candidates.sort(key=lambda p: (len(p), p))
    selected = candidates[0]
    return os.path.dirname(selected), selected


def download_roboflow_dataset(
    api_key: str,
    workspace: str,
    project: str,
    version: int,
    dataset_format: str,
    download_dir: str,
) -> str:
    ensure_dir(download_dir)
    zip_path = os.path.join(
        download_dir,
        f"{workspace}_{project}_v{version}_{dataset_format}.zip",
    )
    if os.path.exists(zip_path) and os.path.getsize(zip_path) > 0:
        log_progress(f"Using cached Roboflow zip: {zip_path}")
        return zip_path

    url = f"https://api.roboflow.com/{workspace}/{project}/{version}/{dataset_format}?api_key={api_key}"
    request = urllib.request.Request(url, headers={"User-Agent": "python-urllib"})
    log_progress(f"Requesting Roboflow export link: {workspace}/{project} v{version} ({dataset_format})")

    with urllib.request.urlopen(request, timeout=180) as response:
        payload = response.read()
        content_type = response.headers.get("Content-Type", "")

    if payload.startswith(b"PK\x03\x04"):
        with open(zip_path, "wb") as f:
            f.write(payload)
        return zip_path

    if "application/json" in content_type or payload[:1] == b"{":
        message = json.loads(payload.decode("utf-8", errors="ignore"))
        if message.get("error"):
            raise RuntimeError(f"Roboflow API error: {message['error']}")

        export_url = (
            message.get("export", {}).get("link")
            or message.get("link")
            or message.get("url")
        )
        if not export_url:
            raise RuntimeError(
                "Roboflow response did not contain a downloadable link. "
                "Check API key/workspace/project/version/format."
            )

        log_progress(f"Downloading Roboflow zip to: {zip_path}")
        download_file(export_url, zip_path, timeout_sec=300, retries=2)
        return zip_path

    raise RuntimeError("Unexpected response from Roboflow API while downloading dataset.")


def extract_zip(zip_path: str, extract_dir: str) -> str:
    clear_dir(extract_dir)
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(extract_dir)
    return extract_dir


def rjina_url(original_url: str) -> str:
    if original_url.startswith("https://r.jina.ai/http://"):
        return original_url
    if original_url.startswith("http://"):
        return "https://r.jina.ai/" + original_url
    if original_url.startswith("https://"):
        return "https://r.jina.ai/http://" + original_url[len("https://") :]
    return "https://r.jina.ai/http://" + original_url


def fetch_text_via_rjina(url: str, timeout_sec: int = 120) -> str:
    request = urllib.request.Request(rjina_url(url), headers={"User-Agent": "python-urllib"})
    with urllib.request.urlopen(request, timeout=timeout_sec) as response:
        payload = response.read()
    return payload.decode("utf-8", errors="ignore")


def infer_state_from_text(
    text: str,
    open_hints: Iterable[str],
    close_hints: Iterable[str],
) -> Optional[int]:
    token = normalize_token(text)
    has_open = False
    has_close = False

    for hint in open_hints:
        hint_token = normalize_token(hint)
        if hint_token and hint_token in token:
            has_open = True
            break

    for hint in close_hints:
        hint_token = normalize_token(hint)
        if hint_token and hint_token in token:
            has_close = True
            break

    if has_close:
        return 0
    if has_open:
        return 1
    return None


def parse_roboflow_browse_entries(
    project_url: str,
    markdown_text: str,
    open_hints: Iterable[str],
    close_hints: Iterable[str],
) -> List[Dict]:
    entries: List[Dict] = []
    seen = set()

    thumb_pattern = re.compile(
        r"https://source\.roboflow\.com/(?P<source>[A-Za-z0-9]+)/(?P<image>[A-Za-z0-9]+)/thumb\.jpg"
    )
    ann_pattern = re.compile(
        r"https://source\.roboflow\.com/(?P<source>[A-Za-z0-9]+)/(?P<image>[A-Za-z0-9]+)/annotation-(?P<ann>[^)\s]+?)\.png"
    )

    for line in markdown_text.splitlines():
        if "source.roboflow.com" not in line or "/thumb.jpg" not in line:
            continue

        thumb_match = thumb_pattern.search(line)
        ann_match = ann_pattern.search(line)
        if not thumb_match or not ann_match:
            continue

        source_id = thumb_match.group("source")
        image_id = thumb_match.group("image")

        if source_id != ann_match.group("source") or image_id != ann_match.group("image"):
            continue

        ann_token = ann_match.group("ann")
        class_state = infer_state_from_text(
            text=ann_token,
            open_hints=open_hints,
            close_hints=close_hints,
        )

        key = (source_id, image_id)
        if key in seen:
            continue
        seen.add(key)

        entries.append(
            {
                "source_id": source_id,
                "image_id": image_id,
                "thumb_url": f"https://source.roboflow.com/{source_id}/{image_id}/thumb.jpg",
                "annotation_url": f"https://source.roboflow.com/{source_id}/{image_id}/annotation-{ann_token}.png",
                "original_url": f"https://source.roboflow.com/{source_id}/{image_id}/original.jpg",
                "image_page_url": f"{project_url.rstrip('/')}/images/{image_id}",
                "class_label": None if class_state is None else int(class_state),
                "annotation_token": ann_token,
            }
        )

    return entries


def extract_roboflow_boxes_from_text(
    text: str,
    open_hints: Iterable[str],
    close_hints: Iterable[str],
) -> List[Dict]:
    records: List[Dict] = []
    seen_keys = set()

    box_pattern = re.compile(
        r'"label"\s*:\s*"(?P<label>[^"]+)"\s*,\s*'
        r'"x"\s*:\s*"?(?P<x>-?\d+(?:\.\d+)?)"?\s*,\s*'
        r'"y"\s*:\s*"?(?P<y>-?\d+(?:\.\d+)?)"?\s*,\s*'
        r'"width"\s*:\s*"?(?P<w>-?\d+(?:\.\d+)?)"?\s*,\s*'
        r'"height"\s*:\s*"?(?P<h>-?\d+(?:\.\d+)?)"?',
        re.IGNORECASE,
    )

    variants = [text]
    decoded_text = text.replace('\\"', '"')
    if decoded_text != text:
        variants.append(decoded_text)

    for variant in variants:
        for match in box_pattern.finditer(variant):
            label_text = match.group("label")
            class_state = infer_state_from_text(
                text=label_text,
                open_hints=open_hints,
                close_hints=close_hints,
            )
            if class_state is None:
                continue

            try:
                x_center = float(match.group("x"))
                y_center = float(match.group("y"))
                width = float(match.group("w"))
                height = float(match.group("h"))
            except Exception:
                continue

            if width <= 1.0 or height <= 1.0:
                continue

            x1 = x_center - width / 2.0
            y1 = y_center - height / 2.0
            x2 = x_center + width / 2.0
            y2 = y_center + height / 2.0

            dedupe_key = (
                int(class_state),
                round(x1, 3),
                round(y1, 3),
                round(x2, 3),
                round(y2, 3),
            )
            if dedupe_key in seen_keys:
                continue
            seen_keys.add(dedupe_key)

            records.append(
                {
                    "class_label": int(class_state),
                    "box_xyxy": (float(x1), float(y1), float(x2), float(y2)),
                    "area": float(max(0.0, width * height)),
                }
            )

    return records


def parse_roboflow_image_page(
    markdown_text: str,
    open_hints: Iterable[str],
    close_hints: Iterable[str],
) -> Tuple[Optional[int], Optional[Tuple[float, float, float, float]]]:
    box_records = extract_roboflow_boxes_from_text(
        text=markdown_text,
        open_hints=open_hints,
        close_hints=close_hints,
    )
    if box_records:
        box_records.sort(key=lambda r: r["area"], reverse=True)
        best = box_records[0]
        return int(best["class_label"]), tuple(best["box_xyxy"])

    label_pattern = re.compile(r'"label"\s*:\s*"(?P<label>[^"]+)"', re.IGNORECASE)
    for variant in (markdown_text, markdown_text.replace('\\"', '"')):
        for match in label_pattern.finditer(variant):
            class_state = infer_state_from_text(
                text=match.group("label"),
                open_hints=open_hints,
                close_hints=close_hints,
            )
            if class_state is not None:
                return int(class_state), None

    return None, None


def estimate_bbox_from_thumb_annotation(
    thumb_image: Image.Image,
    annotation_image: Image.Image,
) -> Optional[Tuple[float, float, float, float]]:
    thumb_rgb = np.asarray(thumb_image.convert("RGB"), dtype=np.int16)
    ann_rgb = np.asarray(annotation_image.convert("RGB"), dtype=np.int16)

    if thumb_rgb.shape != ann_rgb.shape:
        return None

    diff = np.abs(ann_rgb - thumb_rgb).sum(axis=2)
    mask = None

    for threshold in (160, 130, 100, 80, 60):
        candidate = diff >= threshold
        count = int(candidate.sum())
        if count < 20:
            continue
        if count > int(candidate.size * 0.45):
            continue
        mask = candidate
        break

    if mask is None:
        return None

    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None

    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max()) + 1
    y2 = int(ys.max()) + 1

    if x2 - x1 < 3 or y2 - y1 < 3:
        return None

    return float(x1), float(y1), float(x2), float(y2)


def scale_box(
    box: Tuple[float, float, float, float],
    from_size: Tuple[int, int],
    to_size: Tuple[int, int],
) -> Tuple[float, float, float, float]:
    from_w, from_h = from_size
    to_w, to_h = to_size
    if from_w <= 0 or from_h <= 0:
        return 0.0, 0.0, float(max(1, to_w)), float(max(1, to_h))

    sx = float(to_w) / float(from_w)
    sy = float(to_h) / float(from_h)

    x1, y1, x2, y2 = box
    x1 *= sx
    y1 *= sy
    x2 *= sx
    y2 *= sy

    x1 = max(0.0, min(float(max(0, to_w - 1)), x1))
    y1 = max(0.0, min(float(max(0, to_h - 1)), y1))
    x2 = max(x1 + 1.0, min(float(max(1, to_w)), x2))
    y2 = max(y1 + 1.0, min(float(max(1, to_h)), y2))
    return x1, y1, x2, y2


def collect_roboflow_public_entries(
    project_url: str,
    open_hints: Iterable[str],
    close_hints: Iterable[str],
    download_cache_dir: str,
    page_size: int = 50,
    max_images: Optional[int] = None,
    refresh_index: bool = False,
) -> Tuple[List[SampleEntry], Dict]:
    ensure_dir(download_cache_dir)
    open_hints_list = list(open_hints)
    close_hints_list = list(close_hints)

    log_progress(
        f"Roboflow public mode: scanning browse pages from {project_url} "
        f"(page_size={int(page_size)}, max_images={max_images})"
    )

    index_cache_path = os.path.join(download_cache_dir, "browse_index_cache.json")
    browse_pages = 0
    index_cache_used = False
    raw_items: List[Dict] = []

    if not refresh_index and os.path.exists(index_cache_path) and os.path.getsize(index_cache_path) > 0:
        try:
            cached = json.loads(safe_read_text(index_cache_path))
            cache_valid = (
                str(cached.get("project_url", "")).rstrip("/") == str(project_url).rstrip("/")
                and int(cached.get("page_size", -1)) == int(page_size)
                and list(cached.get("open_hints", [])) == open_hints_list
                and list(cached.get("close_hints", [])) == close_hints_list
                and isinstance(cached.get("raw_items"), list)
            )
            if cache_valid and len(cached["raw_items"]) > 0:
                raw_items = cached["raw_items"]
                browse_pages = int(cached.get("browse_pages_scanned", 0))
                index_cache_used = True
                log_progress(
                    f"Using cached browse index: {index_cache_path} "
                    f"(items={len(raw_items)}, pages={browse_pages})"
                )
        except Exception:
            raw_items = []

    if not raw_items:
        collected_raw: Dict[Tuple[str, str], Dict] = {}

        for starting_index in range(0, 100000, page_size):
            browse_url = (
                project_url.rstrip("/")
                + f"/browse?pageSize={int(page_size)}&startingIndex={int(starting_index)}&browseQuery=true"
            )
            markdown = fetch_text_via_rjina(browse_url)
            page_entries = parse_roboflow_browse_entries(
                project_url=project_url,
                markdown_text=markdown,
                open_hints=open_hints_list,
                close_hints=close_hints_list,
            )
            browse_pages += 1

            if not page_entries:
                break

            new_count = 0
            for item in page_entries:
                key = (item["source_id"], item["image_id"])
                if key not in collected_raw:
                    collected_raw[key] = item
                    new_count += 1
            log_progress(
                f"Browse page {browse_pages}: entries={len(page_entries)}, new={new_count}, "
                f"unique_total={len(collected_raw)}"
            )

            if new_count == 0:
                break

            if max_images is not None and len(collected_raw) >= max_images:
                break

            if len(page_entries) < page_size:
                break

        raw_items = list(collected_raw.values())
        raw_items.sort(key=lambda x: x["image_id"])
        try:
            with open(index_cache_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "project_url": project_url.rstrip("/"),
                        "page_size": int(page_size),
                        "open_hints": open_hints_list,
                        "close_hints": close_hints_list,
                        "browse_pages_scanned": int(browse_pages),
                        "raw_items": raw_items,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
        except Exception:
            pass
    else:
        raw_items = sorted(raw_items, key=lambda x: x.get("image_id", ""))

    if max_images is not None:
        raw_items = raw_items[: max(0, int(max_images))]
    log_progress(f"Collected {len(raw_items)} candidate images. Starting downloads and bbox extraction...")

    entries: List[SampleEntry] = []
    failed_downloads = 0
    failed_meta = 0
    failed_class = 0
    failed_bbox = 0
    bbox_from_metadata = 0
    bbox_from_thumb = 0
    metadata_cache_hits = 0
    metadata_fetched = 0
    class_counts = {"0": 0, "1": 0}

    total_items = len(raw_items)
    for index, item in enumerate(raw_items, start=1):
        if index == 1 or index % 50 == 0 or index == total_items:
            log_progress(f"Processing image {index}/{total_items}")
        source_id = item["source_id"]
        image_id = item["image_id"]
        class_label = item.get("class_label")

        source_dir = os.path.join(download_cache_dir, source_id)
        ensure_dir(source_dir)

        original_path = os.path.join(source_dir, f"{image_id}_original.jpg")
        image_meta_path = os.path.join(source_dir, f"{image_id}_meta.json")
        thumb_path = os.path.join(source_dir, f"{image_id}_thumb.jpg")
        ann_path = os.path.join(source_dir, f"{image_id}_annotation.png")

        page_class_label = None
        page_box = None

        if os.path.exists(image_meta_path) and os.path.getsize(image_meta_path) > 0:
            try:
                cached_meta = json.loads(safe_read_text(image_meta_path))

                cached_label = cached_meta.get("class_label")
                if cached_label is not None:
                    page_class_label = int(cached_label)

                cached_box = cached_meta.get("box_xyxy")
                if isinstance(cached_box, (list, tuple)) and len(cached_box) == 4:
                    page_box = tuple(float(v) for v in cached_box)

                metadata_cache_hits += 1
            except Exception:
                page_class_label = None
                page_box = None

        if page_class_label is None or page_box is None:
            try:
                image_page_markdown = fetch_text_via_rjina(item["image_page_url"], timeout_sec=120)
                parsed_class, parsed_box = parse_roboflow_image_page(
                    markdown_text=image_page_markdown,
                    open_hints=open_hints_list,
                    close_hints=close_hints_list,
                )
                metadata_fetched += 1

                if parsed_class is not None:
                    page_class_label = int(parsed_class)
                if parsed_box is not None:
                    page_box = tuple(float(v) for v in parsed_box)

                with open(image_meta_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "image_id": image_id,
                            "class_label": page_class_label,
                            "box_xyxy": None if page_box is None else [float(v) for v in page_box],
                            "image_page_url": item["image_page_url"],
                        },
                        f,
                        ensure_ascii=False,
                        indent=2,
                    )
            except Exception:
                failed_meta += 1

        if page_class_label is not None:
            class_label = int(page_class_label)

        if class_label is None:
            failed_class += 1
            continue
        class_label = int(class_label)

        try:
            if not (os.path.exists(original_path) and os.path.getsize(original_path) > 0):
                download_file(item["original_url"], original_path, timeout_sec=180, retries=2)
        except Exception:
            failed_downloads += 1
            continue

        try:
            with Image.open(original_path) as orig_img:
                orig_w, orig_h = orig_img.size
        except Exception:
            failed_downloads += 1
            continue

        if page_box is not None:
            box = scale_box(page_box, (orig_w, orig_h), (orig_w, orig_h))
            bbox_from_metadata += 1
        else:
            try:
                if not (os.path.exists(thumb_path) and os.path.getsize(thumb_path) > 0):
                    download_file(item["thumb_url"], thumb_path, timeout_sec=180, retries=2)
                if not (os.path.exists(ann_path) and os.path.getsize(ann_path) > 0):
                    download_file(item["annotation_url"], ann_path, timeout_sec=180, retries=2)
            except Exception:
                failed_downloads += 1
                continue

            try:
                with Image.open(thumb_path) as thumb_img, Image.open(ann_path) as ann_img:
                    thumb_w, thumb_h = thumb_img.size
                    thumb_box = estimate_bbox_from_thumb_annotation(thumb_img, ann_img)
            except Exception:
                failed_downloads += 1
                continue

            if thumb_box is None:
                failed_bbox += 1
                # Fallback ROI: centered lower-middle region where drawers usually appear.
                box = (
                    float(orig_w * 0.30),
                    float(orig_h * 0.20),
                    float(orig_w * 0.95),
                    float(orig_h * 0.90),
                )
            else:
                box = scale_box(thumb_box, (thumb_w, thumb_h), (orig_w, orig_h))
                bbox_from_thumb += 1

        entries.append(
            SampleEntry(
                image_path=original_path,
                class_label=class_label,
                box_xyxy=box,
                source="roboflow_public_scrape",
                source_stem=image_id,
            )
        )
        class_counts[str(class_label)] += 1

    stats = {
        "browse_pages_scanned": browse_pages,
        "raw_items_found": len(raw_items),
        "entries_kept": len(entries),
        "failed_downloads": failed_downloads,
        "failed_metadata_fetches": failed_meta,
        "failed_class_inference": failed_class,
        "failed_bbox_fallback_used": failed_bbox,
        "bbox_from_metadata": bbox_from_metadata,
        "bbox_from_thumb_annotation": bbox_from_thumb,
        "metadata_cache_hits": metadata_cache_hits,
        "metadata_fetched": metadata_fetched,
        "class_counts": class_counts,
        "index_cache_used": index_cache_used,
    }
    log_progress(
        f"Roboflow public ingest done: kept={len(entries)}, class0={class_counts['0']}, class1={class_counts['1']}, "
        f"failed_downloads={failed_downloads}, failed_meta={failed_meta}, bbox_fallback={failed_bbox}"
    )
    return entries, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a merged/replaced internal dataset for cash register state detection/classification. "
            "Converts YOLO labels to the project's internal format."
        )
    )

    parser.add_argument("--source-yolo-root", default=None, help="Path to a local YOLO dataset root.")
    parser.add_argument("--source-zip-path", default=None, help="Path to a local zip with YOLO dataset.")
    parser.add_argument("--source-zip-url", default=None, help="Public URL to zip with YOLO dataset.")
    parser.add_argument(
        "--roboflow-public-url",
        default=None,
        help="Public Roboflow Universe project URL (no API key), e.g. https://universe.roboflow.com/workspace/project",
    )
    parser.add_argument("--target-root", default="dataset_merged", help="Output dataset root.")
    parser.add_argument("--mode", choices=["replace", "merge"], default="merge")
    parser.add_argument("--existing-internal-root", default="dataset")
    parser.add_argument("--replace-target", action="store_true", help="Delete target root before writing.")
    parser.add_argument("--no-dedupe", action="store_true", help="Disable duplicate image deduplication.")

    parser.add_argument("--roboflow-api-key", default=None, help="Roboflow API key (optional).")
    parser.add_argument("--workspace", default="shazab-amxv6")
    parser.add_argument("--project", default="cash-register-mezqo")
    parser.add_argument("--version", type=int, default=1)
    parser.add_argument("--format", default="yolov8")
    parser.add_argument("--download-dir", default=os.path.join("external_datasets", "downloads"))
    parser.add_argument("--extract-dir", default=os.path.join("external_datasets", "extracted"))
    parser.add_argument("--rf-page-size", type=int, default=50)
    parser.add_argument("--rf-max-images", type=int, default=None)
    parser.add_argument(
        "--rf-refresh-index",
        action="store_true",
        help="Ignore cached Roboflow browse index and re-fetch it from network.",
    )

    parser.add_argument("--open-hints", nargs="*", default=DEFAULT_OPEN_HINTS)
    parser.add_argument("--close-hints", nargs="*", default=DEFAULT_CLOSE_HINTS)
    parser.add_argument("--open-class-ids", nargs="*", type=int, default=None)
    parser.add_argument("--close-class-ids", nargs="*", type=int, default=None)
    parser.add_argument(
        "--allow-single-class",
        action="store_true",
        help="Do not fail when resulting dataset contains only one class (not recommended).",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_kind = None
    yolo_root = None
    data_yaml_path = None
    class_names = {}
    state_map = {}
    source_stats = {}
    external_entries: List[SampleEntry] = []

    if args.source_yolo_root:
        source_kind = "yolo_root"
        yolo_root = os.path.abspath(args.source_yolo_root)
        if not os.path.isdir(yolo_root):
            raise FileNotFoundError(f"YOLO root not found: {yolo_root}")
        log_progress(f"Using local YOLO dataset: {yolo_root}")
    elif args.source_zip_path or args.source_zip_url:
        source_kind = "yolo_zip"
        if args.source_zip_path:
            zip_path = os.path.abspath(args.source_zip_path)
            if not os.path.isfile(zip_path):
                raise FileNotFoundError(f"Zip file not found: {zip_path}")
            log_progress(f"Using local zip dataset: {zip_path}")
        else:
            ensure_dir(args.download_dir)
            zip_filename = os.path.basename(args.source_zip_url.rstrip("/")) or "dataset.zip"
            if not zip_filename.lower().endswith(".zip"):
                zip_filename += ".zip"
            zip_path = os.path.join(args.download_dir, zip_filename)
            if not (os.path.exists(zip_path) and os.path.getsize(zip_path) > 0):
                log_progress(f"Downloading source zip to: {zip_path}")
                download_file(args.source_zip_url, zip_path, timeout_sec=300, retries=2)
            else:
                log_progress(f"Using cached source zip: {zip_path}")

        extract_root = os.path.join(
            args.extract_dir,
            f"zip_import_{os.path.splitext(os.path.basename(zip_path))[0]}",
        )
        log_progress(f"Extracting zip to: {extract_root}")
        extract_zip(zip_path, extract_root)
        yolo_root, _ = find_first_data_yaml(extract_root)
    elif args.roboflow_public_url:
        source_kind = "roboflow_public_scrape"
        cache_dir = os.path.join(args.download_dir, "roboflow_public_cache")
        external_entries, source_stats = collect_roboflow_public_entries(
            project_url=args.roboflow_public_url,
            open_hints=args.open_hints,
            close_hints=args.close_hints,
            download_cache_dir=cache_dir,
            page_size=max(1, int(args.rf_page_size)),
            max_images=args.rf_max_images,
            refresh_index=args.rf_refresh_index,
        )
    else:
        if not args.roboflow_api_key:
            raise RuntimeError(
                "Provide one of: --source-yolo-root, --source-zip-path, --source-zip-url, "
                "--roboflow-public-url, --roboflow-api-key."
            )

        source_kind = "roboflow_api"
        zip_path = download_roboflow_dataset(
            api_key=args.roboflow_api_key,
            workspace=args.workspace,
            project=args.project,
            version=args.version,
            dataset_format=args.format,
            download_dir=args.download_dir,
        )

        extract_root = os.path.join(
            args.extract_dir,
            f"{args.workspace}_{args.project}_v{args.version}_{args.format}",
        )
        log_progress(f"Extracting Roboflow zip to: {extract_root}")
        extract_zip(zip_path, extract_root)
        yolo_root, _ = find_first_data_yaml(extract_root)

    if source_kind in {"yolo_root", "yolo_zip", "roboflow_api"}:
        log_progress(f"Reading class names from data.yaml under: {yolo_root}")
        _, data_yaml_path = find_first_data_yaml(yolo_root)
        class_names = parse_data_yaml_names(data_yaml_path)
        state_map = infer_state_map(
            names=class_names,
            close_hints=args.close_hints,
            open_hints=args.open_hints,
            close_class_ids=args.close_class_ids,
            open_class_ids=args.open_class_ids,
        )

        if not state_map:
            raise RuntimeError(
                "Could not infer class mapping for open/close states. "
                "Use --open-class-ids and --close-class-ids explicitly."
            )

        external_entries, source_stats = collect_yolo_entries(
            yolo_root=yolo_root,
            state_map=state_map,
            source_tag="external_yolo",
        )
        log_progress(f"Collected external YOLO entries: {len(external_entries)}")

    if not external_entries:
        raise RuntimeError(
            "No usable samples were extracted from external source. "
            "Check class mapping, URLs, and dataset labels."
        )

    entries: List[SampleEntry] = []
    if args.mode == "merge":
        existing_root = os.path.abspath(args.existing_internal_root)
        log_progress(f"Merging with existing internal dataset: {existing_root}")
        existing_entries = collect_internal_entries(existing_root, source_tag="existing_internal")
        log_progress(f"Existing internal entries: {len(existing_entries)}")
        entries.extend(existing_entries)
    entries.extend(external_entries)

    merged_counts = {0: 0, 1: 0}
    for entry in entries:
        merged_counts[int(entry.class_label)] = merged_counts.get(int(entry.class_label), 0) + 1
    log_progress(
        f"Merged class distribution before write: class0={merged_counts.get(0, 0)}, "
        f"class1={merged_counts.get(1, 0)}"
    )

    if not args.allow_single_class:
        if merged_counts.get(0, 0) == 0 or merged_counts.get(1, 0) == 0:
            raise RuntimeError(
                "Resulting dataset contains only one class. "
                "For cash register state classification both classes are required (0=closed, 1=open). "
                "Use another source/merge strategy or pass --allow-single-class to bypass this guard."
            )

    summary = write_internal_dataset(
        entries=entries,
        target_root=os.path.abspath(args.target_root),
        replace_target=args.replace_target,
        dedupe=not args.no_dedupe,
    )

    report = {
        "target_root": os.path.abspath(args.target_root),
        "mode": args.mode,
        "source_kind": source_kind,
        "yolo_root": yolo_root,
        "data_yaml": data_yaml_path,
        "detected_class_names": {str(k): v for k, v in class_names.items()},
        "state_map": {str(k): v for k, v in state_map.items()},
        "external_stats": source_stats,
        "output_summary": summary,
    }

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


