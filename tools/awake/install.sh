#!/bin/bash
set -e

REPO="frorong/sign-model"
BRANCH="master"
INSTALL_DIR="/usr/local/bin"
SCRIPT_NAME="awake"

echo "☕ awake 설치 중..."

# macOS 확인
if [[ "$(uname)" != "Darwin" ]]; then
    echo "❌ 이 도구는 macOS 전용입니다."
    exit 1
fi

# Python3 확인
if ! command -v python3 &> /dev/null; then
    echo "❌ python3가 필요합니다."
    exit 1
fi

# 다운로드
TEMP_FILE=$(mktemp)
curl -fsSL "https://raw.githubusercontent.com/${REPO}/${BRANCH}/tools/awake/awake.py" -o "$TEMP_FILE"

# 설치
if [[ -w "$INSTALL_DIR" ]]; then
    mv "$TEMP_FILE" "${INSTALL_DIR}/${SCRIPT_NAME}"
    chmod +x "${INSTALL_DIR}/${SCRIPT_NAME}"
else
    echo "sudo 권한이 필요합니다..."
    sudo mv "$TEMP_FILE" "${INSTALL_DIR}/${SCRIPT_NAME}"
    sudo chmod +x "${INSTALL_DIR}/${SCRIPT_NAME}"
fi

echo "✅ 설치 완료: ${INSTALL_DIR}/${SCRIPT_NAME}"
echo ""
echo "사용법:"
echo "  awake watch              # 프로세스 감시"
echo "  awake run <command>      # 명령어 실행 동안 잠자기 방지"
echo "  awake status             # 상태 확인"
echo "  awake --help             # 도움말"
