from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from lab_utils.visualization import plot_class_balance, plot_numeric_distribution
SEED = 1234
SPLITS = ('train', 'val', 'test')
LABELS = ('cat', 'dog')
IMAGE_EXTENSIONS = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp')

def list_image_paths_for_group(data_root: Path, split: str, label: str) -> list[Path]:
    label_dir = data_root / split / label
    paths: list[Path] = []
    for pattern in IMAGE_EXTENSIONS:
        paths.extend(label_dir.glob(pattern))
    print(paths)
    return sorted(paths)

def inspect_image_file(path: Path) -> tuple[int, int, float]:
    with Image.open(path) as img:
        img = img.convert('RGB')
        width, height = img.size
        array = np.asarray(img, dtype=np.float32) / 255.0
    return (width, height, float(array.mean()))

def make_metadata_row(path: Path, data_root: Path, split: str, label: str) -> dict[str, object]:
    width, height, mean_intensity = inspect_image_file(path)
    return {'filepath': str(path.relative_to(data_root)), 'label': label, 'split': split, 'width': width, 'height': height, 'mean_intensity': mean_intensity}

def build_metadata_from_folders(data_root: Path) -> pd.DataFrame:
    rows = []
    for split in SPLITS:
        for label in LABELS:
            paths = list_image_paths_for_group(data_root, split, label)
            rows.extend((make_metadata_row(p, data_root, split, label) for p in paths))
    return pd.DataFrame(rows).sort_values(['split', 'label', 'filepath']).reset_index(drop=True)

def load_metadata_table(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)

def summarize_metadata(frame: pd.DataFrame) -> dict[str, object]:
    return {'rows': len(frame), 'columns': frame.columns.tolist(), 'class_counts': frame['label'].value_counts().sort_index(), 'split_counts': frame['split'].value_counts().sort_index()}

def build_label_split_table(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.groupby(['label', 'split']).size().unstack(fill_value=0)

def audit_metadata(frame: pd.DataFrame) -> dict[str, object]:
    return {'missing_values': frame.isna().sum().to_dict(), 'duplicate_filepaths': int(frame['filepath'].duplicated().sum()), 'bad_labels': frame.loc[~frame['label'].isin(['cat', 'dog']), 'label'].unique().tolist(), 'non_positive_sizes': int(((frame['width'] <= 0) | (frame['height'] <= 0)).sum())}

def add_analysis_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result['pixel_count'] = result['width'] * result['height']
    result['aspect_ratio'] = result['width'] / result['height']
    result['brightness_band'] = pd.qcut(result['mean_intensity'], q=4, labels=['darkest', 'dim', 'bright', 'brightest'], duplicates='drop')
    size_reference_pixels = 64 * 64
    result['size_bucket'] = np.select([result['pixel_count'] < size_reference_pixels, result['pixel_count'] == size_reference_pixels, result['pixel_count'] > size_reference_pixels], ['small', 'medium', 'large'], default='unknown')
    return result

def build_split_characteristics_table(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.groupby('split')[['width', 'height', 'pixel_count', 'mean_intensity']].mean().rename(columns={'width': 'avg_width', 'height': 'avg_height', 'pixel_count': 'avg_pixel_count', 'mean_intensity': 'avg_mean_intensity'})

def sample_balanced_by_split_and_label(frame: pd.DataFrame, n_per_group: int, seed: int) -> pd.DataFrame:
    pieces = []
    for _, group in frame.groupby(['split', 'label']):
        pieces.append(group.sample(n=min(n_per_group, len(group)), random_state=seed))
    sampled = pd.concat(pieces, ignore_index=True)
    return sampled.sort_values(['split', 'label', 'filepath']).reset_index(drop=True)
sample_size_per_group = 5
