from __future__ import annotations
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence as torch_pad_sequence
import h5py


class IAMOnDBDataset(Dataset):
    def __init__(self, hdf5_path: str, alphabet: str = None):
        self.hdf5_path = hdf5_path
        
        with h5py.File(hdf5_path, 'r') as f:
            self.strokes = f['strokes'][:]
            self.texts = [t.decode('utf-8') for t in f['texts'][:]]
            self.lengths = f['lengths'][:]
            self.mean = f['mean'][:]
            self.std = f['std'][:]
            self.max_len = f.attrs.get('max_len', 700)
            if 'alphabet' in f.attrs:
                self.alphabet = f.attrs['alphabet']
            else:
                self.alphabet = alphabet or self._build_alphabet()
        
        self.char_to_idx = {c: i for i, c in enumerate(self.alphabet)}
    
    def _build_alphabet(self) -> str:
        chars = set()
        for text in self.texts:
            chars.update(text)
        return ''.join(sorted(chars))
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> dict:
        stroke = self.strokes[idx]
        length = self.lengths[idx]
        text = self.texts[idx]
        
        text_onehot = self._text_to_onehot(text)
        
        return {
            'stroke': torch.tensor(stroke, dtype=torch.float32),
            'text': text,
            'text_onehot': torch.tensor(text_onehot, dtype=torch.float32),
            'text_length': len(text),
            'stroke_length': int(length),
        }
    
    def _text_to_onehot(self, text: str) -> np.ndarray:
        onehot = np.zeros((len(text), len(self.alphabet)), dtype=np.float32)
        for i, char in enumerate(text):
            if char in self.char_to_idx:
                onehot[i, self.char_to_idx[char]] = 1.0
        return onehot
    
    @property
    def alphabet_size(self) -> int:
        return len(self.alphabet)


def collate_fn(batch: list[dict]) -> dict:
    strokes = torch.stack([item['stroke'] for item in batch])
    stroke_lengths = torch.tensor([item['stroke_length'] for item in batch], dtype=torch.long)
    text_lengths = torch.tensor([item['text_length'] for item in batch], dtype=torch.long)
    texts = [item['text'] for item in batch]
    
    text_onehots = [item['text_onehot'] for item in batch]
    text_onehots_padded = torch_pad_sequence(text_onehots, batch_first=True)
    
    return {
        'strokes': strokes,
        'stroke_lengths': stroke_lengths,
        'text_onehots': text_onehots_padded,
        'text_lengths': text_lengths,
        'texts': texts,
    }


def parse_stroke_xml(xml_path: Path) -> list[list[tuple[int, int]]]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    strokes = []
    for stroke_elem in root.findall('.//Stroke'):
        points = []
        for point in stroke_elem.findall('Point'):
            x = int(point.get('x'))
            y = int(point.get('y'))
            points.append((x, y))
        if points:
            strokes.append(points)
    
    return strokes


def parse_text_file(ascii_dir: Path, form_id: str, line_id: str) -> str | None:
    parts = form_id.split('-')
    if len(parts) < 2:
        return None
    
    writer_id = parts[0]
    form_file = ascii_dir / writer_id / f"{form_id}.txt"
    
    if not form_file.exists():
        return None
    
    with open(form_file, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    
    in_csr = False
    for line in lines:
        line = line.strip()
        if line == 'CSR:':
            in_csr = True
            continue
        if in_csr and line:
            if line_id in line or not line.startswith(' '):
                return line
    
    return None


def load_line_texts(ascii_dir: Path) -> dict[str, str]:
    texts = {}
    
    for writer_dir in ascii_dir.iterdir():
        if not writer_dir.is_dir():
            continue
        for form_dir in writer_dir.iterdir():
            if not form_dir.is_dir():
                continue
            for txt_file in form_dir.glob('*.txt'):
                form_id = txt_file.stem
                
                with open(txt_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                lines = content.split('\n')
                in_csr = False
                line_idx = 0
                
                for line in lines:
                    line = line.strip()
                    if line == 'CSR:':
                        in_csr = True
                        continue
                    if in_csr and line:
                        line_key = f"{form_id}-{line_idx:02d}"
                        texts[line_key] = line
                        line_idx += 1
    
    return texts


def load_iam_ondb(data_dir: Path) -> list[dict]:
    samples = []
    
    # 압축 해제 후 구조에 따라 경로 자동 감지
    if (data_dir / 'lineStrokes-all' / 'lineStrokes').exists():
        strokes_dir = data_dir / 'lineStrokes-all' / 'lineStrokes'
        ascii_dir = data_dir / 'ascii-all' / 'ascii'
    else:
        strokes_dir = data_dir / 'lineStrokes'
        ascii_dir = data_dir / 'ascii'
    
    line_texts = {}
    if ascii_dir.exists():
        print(f'Loading text transcriptions from {ascii_dir}...')
        line_texts = load_line_texts(ascii_dir)
        print(f'Loaded {len(line_texts)} text transcriptions')
    
    for writer_dir in strokes_dir.iterdir():
        if not writer_dir.is_dir():
            continue
        for form_dir in writer_dir.iterdir():
            if not form_dir.is_dir():
                continue
            for xml_file in sorted(form_dir.glob('*.xml')):
                strokes = parse_stroke_xml(xml_file)
                if not strokes:
                    continue
                
                filename = xml_file.stem
                parts = filename.split('-')
                if len(parts) >= 3:
                    form_id = f"{parts[0]}-{parts[1]}"
                    line_num = parts[2]
                    line_key = f"{form_id}-{line_num}"
                else:
                    line_key = filename
                
                text = line_texts.get(line_key, '')
                
                samples.append({
                    'strokes': strokes,
                    'text': text,
                    'file': str(xml_file)
                })
    
    return samples
