#!/bin/bash

# MOBISIG 서명 데이터셋 다운로드 스크립트
# 83명 × 45 genuine = 3,735개 서명 (라틴 알파벳)

DATA_DIR="data/mobisig"
ZIP_URL="http://www.ms.sapientia.ro/~manyi/mobisig/MOBISIG.ZIP"

rm -rf "$DATA_DIR"
mkdir -p "$DATA_DIR"

echo "Downloading MOBISIG dataset..."
curl -L -o "$DATA_DIR/MOBISIG.ZIP" "$ZIP_URL"

echo "Extracting..."
unzip -o "$DATA_DIR/MOBISIG.ZIP" -d "$DATA_DIR"

echo "Cleaning up..."
rm -f "$DATA_DIR/MOBISIG.ZIP"

echo ""
echo "Done! Data saved to $DATA_DIR"
echo ""
echo "Dataset info:"
echo "  - 83 users (Latin alphabet names: NAGY, KOVÁCS, TÓTH, etc.)"
echo "  - 45 genuine + 20 forged signatures per user"
echo "  - CSV format: x, y, timestamp, pressure, ..."
echo ""
echo "Next step - convert to H5:"
echo "  python datasets/mobisig.py --data_dir data/mobisig --output data/mobisig.h5"
