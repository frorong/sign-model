# awake

Mac 잠자기 방지 CLI. 특정 프로세스가 실행 중일 때 Mac이 잠자기 모드로 들어가지 않도록 합니다.

## 설치

```bash
curl -fsSL https://raw.githubusercontent.com/frorong/sign-model/master/tools/awake/install.sh | bash
```

## 사용법

### 프로세스 감시
```bash
# 기본 패턴 감시 (train.py, openclaw gateway 등)
awake watch

# 특정 패턴만 감시
awake watch -p "train.py" "my_server"

# 체크 간격 변경 (기본 10초)
awake watch -i 5
```

### 명령어 실행
```bash
# 명령어 실행 동안 잠자기 방지
awake run python train.py --epochs 50
awake run npm run dev
```

### 상태 확인
```bash
awake status
```

### 설정 관리
```bash
# 현재 설정 보기
awake config --show

# 감시 패턴 추가
awake config --add "openclaw gateway"

# 감시 패턴 제거
awake config --remove "node.*server"

# 기본 설정으로 초기화
awake config --init
```

## 설정 파일

`~/.config/awake/config.json`

```json
{
  "patterns": [
    "train.py",
    "openclaw gateway",
    "python.*train",
    "node.*server"
  ],
  "check_interval": 10,
  "caffeinate_options": "-dims"
}
```

## caffeinate 옵션

| 옵션 | 설명 |
|------|------|
| `-d` | 디스플레이 잠자기 방지 |
| `-i` | 시스템 유휴 잠자기 방지 |
| `-m` | 디스크 잠자기 방지 |
| `-s` | 시스템 잠자기 방지 (AC 전원 시) |

## 제거

```bash
sudo rm /usr/local/bin/awake
rm -rf ~/.config/awake
```

## 라이선스

MIT
