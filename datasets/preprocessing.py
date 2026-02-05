import numpy as np


def strokes_to_deltas(strokes: list[list[tuple[int, int]]]) -> np.ndarray:
    all_points = []
    prev_x, prev_y = 0, 0
    
    for stroke_idx, stroke in enumerate(strokes):
        for point_idx, (x, y) in enumerate(stroke):
            dx = x - prev_x
            dy = y - prev_y
            eos = 1.0 if point_idx == len(stroke) - 1 else 0.0
            all_points.append([dx, dy, eos])
            prev_x, prev_y = x, y
    
    return np.array(all_points, dtype=np.float32)


def compute_statistics(all_deltas: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    all_dx = []
    all_dy = []
    
    for deltas in all_deltas:
        all_dx.extend(deltas[:, 0].tolist())
        all_dy.extend(deltas[:, 1].tolist())
    
    dx_array = np.array(all_dx, dtype=np.float32)
    dy_array = np.array(all_dy, dtype=np.float32)
    
    mean = np.array([dx_array.mean(), dy_array.mean()], dtype=np.float32)
    std = np.array([dx_array.std(), dy_array.std()], dtype=np.float32)
    std = np.where(std == 0, 1.0, std)
    
    return mean, std


def standardize_strokes(strokes: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    normalized = strokes.copy()
    normalized[:, 0] = (strokes[:, 0] - mean[0]) / std[0]
    normalized[:, 1] = (strokes[:, 1] - mean[1]) / std[1]
    return normalized


def pad_sequence(strokes: np.ndarray, max_len: int) -> tuple[np.ndarray, int]:
    actual_len = len(strokes)
    
    if actual_len >= max_len:
        return strokes[:max_len], max_len
    
    padding = np.zeros((max_len - actual_len, 3), dtype=np.float32)
    return np.concatenate([strokes, padding], axis=0), actual_len


def build_alphabet(texts: list[str]) -> str:
    chars = set()
    for text in texts:
        chars.update(text)
    return ''.join(sorted(chars))


def text_to_onehot(text: str, alphabet: str) -> np.ndarray:
    char_to_idx = {c: i for i, c in enumerate(alphabet)}
    onehot = np.zeros((len(text), len(alphabet)), dtype=np.float32)
    
    for i, char in enumerate(text):
        if char in char_to_idx:
            onehot[i, char_to_idx[char]] = 1.0
    
    return onehot


def filter_by_length(samples: list[dict], min_len: int = 10, max_len: int = 700) -> list[dict]:
    filtered = []
    for sample in samples:
        deltas = strokes_to_deltas(sample['strokes'])
        if min_len <= len(deltas) <= max_len:
            filtered.append(sample)
    return filtered
