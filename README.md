# AI Signature Synthesis Model

LSTM 기반 손글씨 합성 모델 (Alex Graves의 논문 구현)

## 설치

```bash
pip install -r requirements.txt
```

## 사용법

### 1. 데이터 다운로드 및 압축 해제

[IAM-OnDB](https://fki.tic.heia-fr.ch/databases/iam-on-line-handwriting-database)에서 다음 파일 다운로드:
- `lineStrokes-all.tar.gz` - stroke 좌표 데이터
- `ascii-all.tar.gz` - 텍스트 transcription

다운로드한 파일을 `data/raw/`에 배치한 후 압축 해제:

```bash
cd data/raw
tar -xzf lineStrokes-all.tar.gz
tar -xzf ascii-all.tar.gz
```

압축 해제 후 구조:
```
data/raw/
├── lineStrokes-all/
│   └── lineStrokes/
└── ascii-all/
    └── ascii/
```

### 2. 데이터 전처리

```bash
python scripts/prepare_data.py \
    --data_dir ./data/raw \
    --output_dir ./data \
    --max_len 700 \
    --min_len 10 \
    --train_split 0.9
```

생성되는 파일:
- `data/train.h5` - 학습 데이터
- `data/val.h5` - 검증 데이터
- `data/stats.npz` - 정규화 통계 (mean, std, alphabet)

### 3. 학습

```bash
python scripts/train.py --config configs/default.yaml --train_data ./data/train.h5 --val_data ./data/val.h5 --epochs 50
```

### 4. 합성

```bash
python scripts/synthesize.py --model checkpoints/model_final.pt --config configs/default.yaml --text "Hello World" --output hello.svg
```

## 데이터 파이프라인

### IAM-OnDB 디렉토리 구조

```
iam_ondb_home/
├── lineStrokes-all/
│   └── lineStrokes/
│       └── {writer}/{form}/{line}.xml  # stroke 좌표
└── ascii-all/
    └── ascii/
        └── {writer}/{form}.txt         # 텍스트 transcription
```

### 전처리 과정

1. **XML 파싱**: 각 stroke의 (x, y) 좌표 추출
2. **상대 좌표 변환**: (x, y) → (dx, dy, eos)
   - dx, dy: 이전 포인트로부터의 상대 오프셋
   - eos: 각 stroke의 끝에서 1, 그 외 0
3. **표준화**: 전체 데이터셋의 mean/std로 정규화
4. **패딩**: max_len까지 zero-padding

### HDF5 포맷

```python
{
    'strokes': (N, max_len, 3),   # (dx, dy, eos) 정규화된 상대 좌표
    'texts': (N,),                 # 텍스트 transcription
    'lengths': (N,),               # 실제 시퀀스 길이
    'mean': (2,),                  # 정규화 mean (dx, dy)
    'std': (2,),                   # 정규화 std (dx, dy)
    attrs['alphabet']: str,        # 알파벳 문자열
    attrs['max_len']: int,         # 최대 시퀀스 길이
}
```

## 프로젝트 구조

```
sign-model/
├── configs/          # 학습 설정
├── data/             # 전처리된 데이터 (gitignore)
├── datasets/         # 데이터셋 처리
│   ├── iam_ondb.py       # IAM-OnDB 로더, Dataset 클래스
│   └── preprocessing.py  # 좌표 변환, 정규화
├── models/           # 모델 구현
├── training/         # 학습 로직
├── inference/        # 추론 및 ONNX 변환
└── scripts/          # CLI 스크립트
```

## 참조

- [Generating Sequences With RNNs](https://arxiv.org/abs/1308.0850) (Alex Graves, 2013)
- [IAM-OnDB](https://fki.tic.heia-fr.ch/databases/iam-on-line-handwriting-database)
