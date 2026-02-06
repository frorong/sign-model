from __future__ import annotations
import numpy as np
from pathlib import Path
from typing import Optional
import h5py
import csv


MOBISIG_NAMES = [
    "", "NAGY", "KOVÁCS", "TÓTH", "SZABÓ", "HORVÁTH", "VARGA", "KISS", "MOLNÁR",
    "NÉMETH", "FARKAS", "BALOGH", "PAPP", "TAKÁCS", "JUHÁSZ", "LAKATOS", "MÉSZÁROS",
    "OLÁH", "SIMON", "RÁCZ", "FEKETE", "SZILÁGYI", "TÖRÖK", "FEHÉR", "BALÁZS",
    "GÁL", "KIS", "SZŰCS", "KOCSIS", "PINTÉR", "FODOR", "ORSÓS", "SZALAI",
    "SIPOS", "MAGYAR", "LUKÁCS", "GULYÁS", "BIRÓ", "KIRÁLY", "KATONA", "LÁSZLÓ",
    "JAKAB", "BOGDÁN", "BALOG", "SÁNDOR", "BOROS", "FAZEKAS", "KELEMEN", "ANTAL",
    "SOMOGYI", "VÁRADI", "FÜLÖP", "OROSZ", "VINCZE", "VERES", "HEGEDŰS", "DEÁK",
    "BUDAI", "PAP", "BÁLINT", "PÁL", "ILLÉS", "SZŐKE", "VÖRÖS", "VASS", "BOGNÁR",
    "LENGYEL", "FÁBIÁN", "BODNÁR", "SZÜCS", "HAJDU", "HALÁSZ", "JÓNÁS", "KOZMA",
    "MÁTÉ", "SZÉKELY", "GÁSPÁR", "PÁSZTOR", "BAKOS", "DUDÁS", "MAJOR", "VIRÁG",
    "ORBÁN", "NOVÁK"
]


def load_mobisig(data_dir: str | Path, genuine_only: bool = True) -> tuple[list[np.ndarray], list[str], list[str]]:
    """
    MOBISIG 데이터 로드
    
    CSV 포맷: x, y, timestamp, pressure, finger_area, vx, vy, ax, ay, az, gx, gy, gz
    
    Returns:
        strokes: 각 서명의 스트로크 좌표 리스트
        user_ids: 각 서명의 사용자 ID
        names: 각 서명에 해당하는 이름 (라틴 알파벳)
    """
    data_dir = Path(data_dir)
    
    strokes = []
    user_ids = []
    names = []
    
    for user_dir in sorted(data_dir.glob("USER*")):
        user_id = user_dir.name.replace("USER", "")
        user_num = int(user_id)
        name = MOBISIG_NAMES[user_num] if user_num < len(MOBISIG_NAMES) else f"USER{user_num}"
        
        for sig_file in sorted(user_dir.glob("*.csv")):
            filename = sig_file.stem
            
            if genuine_only and "_FOR_" in filename:
                continue
            
            stroke_data = parse_mobisig_file(sig_file)
            if stroke_data is None or len(stroke_data) < 10:
                continue
            
            strokes.append(stroke_data)
            user_ids.append(user_id)
            names.append(name)
    
    print(f"Loaded {len(strokes)} signatures from {len(set(user_ids))} users")
    
    return strokes, user_ids, names


def parse_mobisig_file(filepath: Path) -> Optional[np.ndarray]:
    """
    MOBISIG CSV 파일 파싱
    포맷: x, y, timestamp, pressure, finger_area, vx, vy, ax, ay, az, gx, gy, gz
    """
    points = []
    prev_pressure = 0
    
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 4:
                continue
            
            try:
                x = float(row[0])
                y = float(row[1])
                pressure = float(row[3]) if len(row) > 3 else 1.0
            except ValueError:
                continue
            
            eos = 1.0 if (prev_pressure > 0.1 and pressure < 0.1) else 0.0
            prev_pressure = pressure
            
            if pressure > 0.05:
                points.append([x, y, eos])
    
    if len(points) < 2:
        return None
    
    return np.array(points, dtype=np.float32)


def convert_to_relative(strokes: np.ndarray) -> np.ndarray:
    """절대 좌표를 상대 좌표(dx, dy, eos)로 변환"""
    coords = strokes[:, :2]
    eos = strokes[:, 2:3]
    
    dx_dy = np.diff(coords, axis=0, prepend=coords[:1])
    dx_dy[0] = 0
    
    return np.concatenate([dx_dy, eos], axis=1).astype(np.float32)


def normalize_signature(strokes: np.ndarray) -> np.ndarray:
    """서명 정규화 (위치, 스케일)"""
    coords = strokes[:, :2].copy()
    
    coords = coords - coords.mean(axis=0)
    
    std = coords.std()
    if std > 1e-6:
        coords = coords / std
    
    result = strokes.copy()
    result[:, :2] = coords
    return result


def augment_signature(strokes: np.ndarray, 
                      scale_range: tuple = (0.8, 1.2),
                      rotation_range: tuple = (-10, 10),
                      noise_std: float = 0.02) -> np.ndarray:
    """서명 데이터 증강"""
    result = strokes.copy()
    coords = result[:, :2].copy()
    
    scale = np.random.uniform(*scale_range)
    coords = coords * scale
    
    angle = np.radians(np.random.uniform(*rotation_range))
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])
    coords = coords @ rotation_matrix.T
    
    if noise_std > 0:
        coords = coords + np.random.randn(*coords.shape) * noise_std
    
    result[:, :2] = coords
    return result


def prepare_mobisig_h5(data_dir: str | Path, 
                       output_path: str | Path,
                       max_seq_len: int = 500,
                       augment_factor: int = 5,
                       genuine_only: bool = True) -> None:
    """MOBISIG를 H5 파일로 변환"""
    strokes_list, user_ids, names = load_mobisig(data_dir, genuine_only)
    
    if len(strokes_list) == 0:
        print("Error: No signatures loaded")
        return
    
    all_strokes = []
    all_user_ids = []
    all_names = []
    
    for strokes, user_id, name in zip(strokes_list, user_ids, names):
        strokes_rel = convert_to_relative(strokes)
        strokes_norm = normalize_signature(strokes_rel)
        
        if len(strokes_norm) > max_seq_len:
            strokes_norm = strokes_norm[:max_seq_len]
        
        all_strokes.append(strokes_norm)
        all_user_ids.append(user_id)
        all_names.append(name)
        
        for _ in range(augment_factor - 1):
            strokes_aug = augment_signature(strokes_rel.copy())
            strokes_aug = normalize_signature(strokes_aug)
            
            if len(strokes_aug) > max_seq_len:
                strokes_aug = strokes_aug[:max_seq_len]
            
            all_strokes.append(strokes_aug)
            all_user_ids.append(user_id)
            all_names.append(name)
    
    unique_users = sorted(set(all_user_ids), key=int)
    user_to_idx = {u: i for i, u in enumerate(unique_users)}
    user_indices = [user_to_idx[u] for u in all_user_ids]
    
    max_len = max(len(s) for s in all_strokes)
    padded_strokes = np.zeros((len(all_strokes), max_len, 3), dtype=np.float32)
    lengths = np.zeros(len(all_strokes), dtype=np.int32)
    
    for i, s in enumerate(all_strokes):
        padded_strokes[i, :len(s)] = s
        lengths[i] = len(s)
    
    mean = np.zeros(2, dtype=np.float32)
    total_points = 0
    
    for s, l in zip(padded_strokes, lengths):
        mean += s[:l, :2].sum(axis=0)
        total_points += l
    mean /= total_points
    
    var = np.zeros(2, dtype=np.float32)
    for s, l in zip(padded_strokes, lengths):
        var += ((s[:l, :2] - mean) ** 2).sum(axis=0)
    std = np.sqrt(var / total_points)
    std = np.maximum(std, 1e-6)
    
    for i, l in enumerate(lengths):
        padded_strokes[i, :l, :2] = (padded_strokes[i, :l, :2] - mean) / std
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('strokes', data=padded_strokes)
        f.create_dataset('lengths', data=lengths)
        f.create_dataset('user_indices', data=np.array(user_indices, dtype=np.int32))
        f.create_dataset('mean', data=mean)
        f.create_dataset('std', data=std)
        f.attrs['num_users'] = len(unique_users)
        f.attrs['genuine_only'] = genuine_only
        
        user_ids_encoded = [u.encode('utf-8') for u in unique_users]
        f.create_dataset('user_ids', data=user_ids_encoded)
        
        names_encoded = [n.encode('utf-8') for n in all_names]
        f.create_dataset('names', data=names_encoded)
    
    print(f"\nSaved {len(all_strokes)} signatures from {len(unique_users)} users to {output_path}")
    print(f"  Max sequence length: {max_len}")
    print(f"  Mean: {mean}, Std: {std}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='MOBISIG 디렉토리')
    parser.add_argument('--output', type=str, default='data/mobisig.h5')
    parser.add_argument('--augment', type=int, default=5)
    parser.add_argument('--max_len', type=int, default=500)
    parser.add_argument('--include_forged', action='store_true')
    args = parser.parse_args()
    
    prepare_mobisig_h5(
        args.data_dir, 
        args.output, 
        max_seq_len=args.max_len,
        augment_factor=args.augment,
        genuine_only=not args.include_forged
    )
