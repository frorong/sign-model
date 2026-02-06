# Data Quality Report

Dataset: `data/train.h5`

## Statistics

- Samples: 6048
- Alphabet size: 79

### Stroke Length
- Range: [113, 700]
- Mean: 556.4 ± 91.8

### Text Length
- Range: [1, 51]
- Mean: 27.6 ± 7.0

## Anomalies

| Type | Count |
|------|-------|
| too_short | 0 |
| too_long | 2239 |
| extreme_steps_per_char | 110 |
| large_jumps | 6048 |
| static_strokes | 0 |
| extreme_pen_up | 24 |
