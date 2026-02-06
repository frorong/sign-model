# 온라인 손글씨/서명 데이터셋

## 현재 사용 중

### MobiSig
- **파일**: `data/mobisig.h5` (108MB)
- **샘플 수**: 미확인
- **특징**: 모바일 서명 데이터

### Train/Val Split
- **train.h5**: 6,048 샘플, 49MB
- **val.h5**: ~670 샘플, 5.4MB
- **시퀀스 길이**: 최대 700

---

## 추가 가능한 데이터셋

### 1. IAM On-Line Handwriting Database (IAM-OnDB)
- **URL**: https://fki.tic.heia-fr.ch/databases/iam-on-line-handwriting-database
- **규모**: 221명, 13,049 라인, 86,272 단어
- **형식**: XML (stroke x, y, time)
- **언어**: 영어
- **라이선스**: 비상업적 연구용 무료
- **등록 필요**: 예

**다운로드 파일:**
- `lineStrokes-all.tar.gz` — 라인별 온라인 스트로크
- `original-xml-all.tar.gz` — 원본 폼
- `ascii-all.tar.gz` — 텍스트 전사

### 2. SVC2021 (Signature Verification Competition)
- **URL**: https://github.com/BiDAlab/SVC2021_EvalDB
- **규모**: 복수 시나리오 (office, mobile)
- **형식**: 좌표 시퀀스
- **라이선스**: 라이선스 계약 필요 (atvs@uam.es)

### 3. MCYT Baseline Corpus
- **특징**: 다중 모달 생체인식 DB
- **포함**: 서명, 지문
- **라이선스**: 학술 연구용

### 4. BiosecurID
- **규모**: 400명
- **포함**: 온라인/오프라인 서명 + 7개 생체정보
- **형식**: 동적 신호 (x, y, pressure, time)

### 5. BiosecurID-SONOF DB
- **규모**: 132명
- **내용**: 진짜 서명 16개 + 위조 서명 12개 per user
- **특징**: 합성 서명 포함

---

## 데이터 전처리 파이프라인

### IAM-OnDB XML → H5 변환 예시

```python
import xml.etree.ElementTree as ET
import numpy as np
import h5py

def parse_iam_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    strokes = []
    for stroke in root.findall('.//Stroke'):
        points = []
        for point in stroke.findall('Point'):
            x = float(point.get('x'))
            y = float(point.get('y'))
            points.append([x, y, 0])  # eos=0
        if points:
            points[-1][2] = 1  # 마지막에 pen-up
            strokes.extend(points)
    
    return np.array(strokes, dtype=np.float32)
```

### 데이터 병합 시 주의사항

1. **좌표 정규화**: 각 데이터셋의 좌표 스케일이 다름
2. **샘플링 레이트**: 시간 간격 통일 필요
3. **알파벳 통합**: 문자셋 합집합으로 확장
4. **품질 필터링**: 이상치 제거 (너무 짧거나 긴 시퀀스)

---

## 다음 단계

1. [ ] IAM-OnDB 등록 및 다운로드
2. [ ] XML 파서 구현
3. [ ] 좌표 정규화 통일
4. [ ] 기존 데이터와 병합
5. [ ] 통합 데이터로 재학습
