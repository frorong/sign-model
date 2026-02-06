# Data Quality Report

Dataset: `data/train_words.h5`

## Statistics

- Samples: 23479
- Alphabet size: 79

### Stroke Length
- Range: [10, 697]
- Mean: 116.2 ± 82.2

### Text Length
- Range: [2, 20]
- Mean: 4.8 ± 2.5

## Anomalies

| Type | Count |
|------|-------|
| too_short | 0 |
| too_long | 56 |
| extreme_steps_per_char | 432 |
| large_jumps | 19118 |
| static_strokes | 0 |
| extreme_pen_up | 244 |
