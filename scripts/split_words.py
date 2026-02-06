#!/usr/bin/env python3
"""문장 데이터를 단어 단위로 분리하여 새 데이터셋 생성"""
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import h5py
import numpy as np
from tqdm import tqdm


def find_word_boundaries(stroke: np.ndarray, num_spaces: int, 
                         min_jump: float = 3.0) -> list[int]:
    """pen-up 후 x 점프로 단어 경계 추정"""
    pen_ups = stroke[:, 2] > 0.5
    pen_up_indices = np.where(pen_ups)[0]
    
    jumps = []
    for pu_idx in pen_up_indices:
        if pu_idx + 1 < len(stroke):
            x_jump = stroke[pu_idx + 1, 0]
            if x_jump > min_jump:
                jumps.append((pu_idx, x_jump))
    
    jumps.sort(key=lambda x: -x[1])
    
    boundaries = sorted([j[0] for j in jumps[:num_spaces]])
    return boundaries


def split_stroke_by_boundaries(stroke: np.ndarray, boundaries: list[int]) -> list[np.ndarray]:
    """경계 위치로 스트로크 분리"""
    if not boundaries:
        return [stroke]
    
    segments = []
    start = 0
    
    for boundary in boundaries:
        if boundary > start:
            segment = stroke[start:boundary + 1].copy()
            segment[-1, 2] = 1.0
            segments.append(segment)
        start = boundary + 1
    
    if start < len(stroke):
        segments.append(stroke[start:].copy())
    
    return segments


def process_dataset(input_path: str, output_path: str, 
                    min_word_len: int = 2, max_word_len: int = 20,
                    min_stroke_len: int = 10):
    """데이터셋 처리"""
    
    with h5py.File(input_path, 'r') as f:
        strokes = f['strokes'][:]
        texts = [t.decode('utf-8') if isinstance(t, bytes) else t for t in f['texts'][:]]
        lengths = f['lengths'][:]
        alphabet = f.attrs['alphabet']
        mean = f['mean'][:]
        std = f['std'][:]
    
    word_strokes = []
    word_texts = []
    word_lengths = []
    
    skipped_mismatch = 0
    skipped_short = 0
    
    for idx in tqdm(range(len(texts)), desc="Processing"):
        text = texts[idx]
        length = lengths[idx]
        stroke = strokes[idx, :length]
        
        words = text.split()
        num_spaces = len(words) - 1
        
        if num_spaces == 0:
            if len(text) >= min_word_len and length >= min_stroke_len:
                word_strokes.append(stroke)
                word_texts.append(text)
                word_lengths.append(length)
            continue
        
        boundaries = find_word_boundaries(stroke, num_spaces)
        
        if len(boundaries) != num_spaces:
            skipped_mismatch += 1
            continue
        
        segments = split_stroke_by_boundaries(stroke, boundaries)
        
        if len(segments) != len(words):
            skipped_mismatch += 1
            continue
        
        for word, segment in zip(words, segments):
            word_clean = word.strip('.,!?;:\'"()[]{}')
            
            if len(word_clean) < min_word_len or len(word_clean) > max_word_len:
                skipped_short += 1
                continue
            
            if len(segment) < min_stroke_len:
                skipped_short += 1
                continue
            
            word_strokes.append(segment)
            word_texts.append(word_clean)
            word_lengths.append(len(segment))
    
    print(f"\n=== 결과 ===")
    print(f"원본 샘플: {len(texts)}")
    print(f"생성된 단어: {len(word_strokes)}")
    print(f"경계 불일치로 스킵: {skipped_mismatch}")
    print(f"길이 조건 미달로 스킵: {skipped_short}")
    
    max_len = max(word_lengths) if word_lengths else 0
    max_len = ((max_len // 100) + 1) * 100
    
    n_samples = len(word_strokes)
    strokes_padded = np.zeros((n_samples, max_len, 3), dtype=np.float32)
    
    for i, (stroke, length) in enumerate(zip(word_strokes, word_lengths)):
        strokes_padded[i, :length] = stroke
    
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('strokes', data=strokes_padded, compression='gzip')
        f.create_dataset('texts', data=np.array(word_texts, dtype='S'))
        f.create_dataset('lengths', data=np.array(word_lengths, dtype=np.int32))
        f.create_dataset('mean', data=mean)
        f.create_dataset('std', data=std)
        f.attrs['alphabet'] = alphabet
        f.attrs['max_len'] = max_len
    
    print(f"\n저장됨: {output_path}")
    print(f"Max length: {max_len}")
    
    text_lens = [len(t) for t in word_texts]
    print(f"\n=== 단어 길이 분포 ===")
    print(f"Min: {min(text_lens)}, Max: {max(text_lens)}, Mean: {np.mean(text_lens):.1f}")


def main():
    parser = argparse.ArgumentParser(description='Split sentence data into words')
    parser.add_argument('--input', type=str, default='data/train.h5')
    parser.add_argument('--output', type=str, default='data/train_words.h5')
    parser.add_argument('--min_word_len', type=int, default=2)
    parser.add_argument('--max_word_len', type=int, default=20)
    parser.add_argument('--min_stroke_len', type=int, default=10)
    args = parser.parse_args()
    
    process_dataset(
        args.input, 
        args.output,
        args.min_word_len,
        args.max_word_len,
        args.min_stroke_len
    )


if __name__ == '__main__':
    main()
