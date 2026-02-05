import numpy as np
import torch
from torch.utils.data import Dataset


class SyntheticStrokeDataset(Dataset):
    """
    간단한 합성 stroke 데이터셋 (파이프라인 검증용)
    
    각 샘플은 텍스트에 대응하는 간단한 stroke 시퀀스를 생성.
    실제 손글씨가 아니라 단순 패턴이지만, 학습 파이프라인 테스트에는 충분.
    """
    
    def __init__(self, num_samples: int = 1000, max_len: int = 200, alphabet: str = None):
        self.num_samples = num_samples
        self.max_len = max_len
        self.alphabet = alphabet or "abcdefghijklmnopqrstuvwxyz "
        self.words = ["hello", "world", "test", "sign", "name", "alex", "john", "kate"]
        
        self.data = [self._generate_sample() for _ in range(num_samples)]
    
    def _generate_sample(self) -> dict:
        text = np.random.choice(self.words)
        strokes = self._text_to_strokes(text)
        return {'stroke': strokes, 'text': text, 'length': len(strokes)}
    
    def _text_to_strokes(self, text: str) -> np.ndarray:
        strokes = []
        x, y = 0.0, 0.0
        
        for i, char in enumerate(text):
            char_strokes = self._char_to_strokes(char)
            strokes.extend(char_strokes)
            x += 1.0
        
        strokes = np.array(strokes, dtype=np.float32)
        
        if len(strokes) > self.max_len:
            strokes = strokes[:self.max_len]
        elif len(strokes) < self.max_len:
            padding = np.zeros((self.max_len - len(strokes), 3), dtype=np.float32)
            strokes = np.concatenate([strokes, padding], axis=0)
        
        return strokes
    
    def _char_to_strokes(self, char: str) -> list:
        strokes = []
        num_points = np.random.randint(10, 25)
        
        for j in range(num_points):
            dx = np.random.randn() * 0.1 + 0.05
            dy = np.random.randn() * 0.2
            eos = 1.0 if j == num_points - 1 else 0.0
            strokes.append([dx, dy, eos])
        
        return strokes
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> dict:
        sample = self.data[idx]
        return {
            'stroke': torch.tensor(sample['stroke'], dtype=torch.float32),
            'text': sample['text'],
            'length': sample['length']
        }


def collate_fn(batch: list) -> dict:
    strokes = torch.stack([b['stroke'] for b in batch])
    texts = [b['text'] for b in batch]
    lengths = torch.tensor([b['length'] for b in batch])
    return {'stroke': strokes, 'text': texts, 'length': lengths}
