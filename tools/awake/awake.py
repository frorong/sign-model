#!/usr/bin/env python3
"""
awake - Mac 잠자기 방지 CLI

특정 프로세스가 실행 중일 때 Mac이 잠자기 모드로 들어가지 않도록 합니다.
"""
from __future__ import annotations

import argparse
import subprocess
import time
import signal
import sys
import json
from pathlib import Path
from typing import Optional


DEFAULT_CONFIG = {
    "patterns": [
        "train.py",
        "openclaw gateway",
        "python.*train",
        "node.*server"
    ],
    "check_interval": 10,
    "caffeinate_options": "-dims"
}

CONFIG_PATH = Path.home() / ".config" / "awake" / "config.json"


class AwakeDaemon:
    def __init__(self, patterns: list[str], interval: int = 10, options: str = "-dims"):
        self.patterns = patterns
        self.interval = interval
        self.options = options
        self.caffeinate_proc = None
        self.running = True
        
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)
    
    def _handle_signal(self, signum, frame):
        print("\n종료 중...")
        self.running = False
        self._stop_caffeinate()
        sys.exit(0)
    
    def _check_processes(self) -> list[str]:
        matched = []
        for pattern in self.patterns:
            result = subprocess.run(
                ["pgrep", "-f", pattern],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                pids = result.stdout.strip().split('\n')
                matched.append(f"{pattern} (PIDs: {', '.join(pids)})")
        return matched
    
    def _start_caffeinate(self):
        if self.caffeinate_proc is None:
            self.caffeinate_proc = subprocess.Popen(
                ["caffeinate"] + self.options.split(),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            print(f"☕ caffeinate 활성화 (PID: {self.caffeinate_proc.pid})")
    
    def _stop_caffeinate(self):
        if self.caffeinate_proc is not None:
            self.caffeinate_proc.terminate()
            self.caffeinate_proc.wait()
            self.caffeinate_proc = None
            print("😴 caffeinate 비활성화")
    
    def run(self):
        print(f"👀 감시 중: {', '.join(self.patterns)}")
        print(f"⏱  체크 간격: {self.interval}초")
        print("Ctrl+C로 종료\n")
        
        was_active = False
        first_check = True
        
        while self.running:
            matched = self._check_processes()
            
            if matched:
                if not was_active:
                    print(f"✅ 감지됨: {', '.join(matched)}")
                    self._start_caffeinate()
                    was_active = True
            else:
                if was_active:
                    print("❌ 감시 대상 프로세스 없음")
                    self._stop_caffeinate()
                    was_active = False
                elif first_check:
                    print("⏳ 대기 중... 감시 대상 프로세스 없음")
            
            first_check = False
            time.sleep(self.interval)


def load_config() -> dict:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            return json.load(f)
    return DEFAULT_CONFIG


def save_config(config: dict):
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, 'w') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"설정 저장됨: {CONFIG_PATH}")


def cmd_watch(args):
    config = load_config()
    patterns = args.patterns if args.patterns else config["patterns"]
    interval = args.interval if args.interval else config["check_interval"]
    options = config.get("caffeinate_options", "-dims")
    
    daemon = AwakeDaemon(patterns, interval, options)
    daemon.run()


def cmd_run(args):
    """특정 명령어 실행 동안 잠자기 방지"""
    command = " ".join(args.command)
    print(f"☕ 실행: {command}")
    print("잠자기 방지 활성화\n")
    
    result = subprocess.run(
        ["caffeinate", "-dims"] + args.command
    )
    sys.exit(result.returncode)


def cmd_status(args):
    result = subprocess.run(["pgrep", "-x", "caffeinate"], capture_output=True, text=True)
    if result.returncode == 0:
        pids = result.stdout.strip().split('\n')
        print(f"☕ caffeinate 실행 중 (PIDs: {', '.join(pids)})")
    else:
        print("😴 caffeinate 미실행")


def cmd_config(args):
    if args.show:
        config = load_config()
        print(json.dumps(config, indent=2, ensure_ascii=False))
    elif args.add:
        config = load_config()
        if args.add not in config["patterns"]:
            config["patterns"].append(args.add)
            save_config(config)
            print(f"추가됨: {args.add}")
        else:
            print(f"이미 존재: {args.add}")
    elif args.remove:
        config = load_config()
        if args.remove in config["patterns"]:
            config["patterns"].remove(args.remove)
            save_config(config)
            print(f"제거됨: {args.remove}")
        else:
            print(f"존재하지 않음: {args.remove}")
    elif args.init:
        save_config(DEFAULT_CONFIG)


def main():
    parser = argparse.ArgumentParser(
        description="Mac 잠자기 방지 CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  awake watch                    # 설정된 패턴 감시
  awake watch -p "train.py"      # 특정 패턴만 감시
  awake run python train.py      # 명령어 실행 동안 잠자기 방지
  awake status                   # caffeinate 상태 확인
  awake config --show            # 설정 보기
  awake config --add "my_script" # 패턴 추가
"""
    )
    
    subparsers = parser.add_subparsers(dest="command", help="명령어")
    
    watch_parser = subparsers.add_parser("watch", help="프로세스 감시 및 잠자기 방지")
    watch_parser.add_argument("-p", "--patterns", nargs="+", help="감시할 프로세스 패턴")
    watch_parser.add_argument("-i", "--interval", type=int, help="체크 간격 (초)")
    watch_parser.set_defaults(func=cmd_watch)
    
    run_parser = subparsers.add_parser("run", help="명령어 실행 동안 잠자기 방지")
    run_parser.add_argument("command", nargs="+", help="실행할 명령어")
    run_parser.set_defaults(func=cmd_run)
    
    status_parser = subparsers.add_parser("status", help="caffeinate 상태 확인")
    status_parser.set_defaults(func=cmd_status)
    
    config_parser = subparsers.add_parser("config", help="설정 관리")
    config_parser.add_argument("--show", action="store_true", help="현재 설정 보기")
    config_parser.add_argument("--add", help="감시 패턴 추가")
    config_parser.add_argument("--remove", help="감시 패턴 제거")
    config_parser.add_argument("--init", action="store_true", help="기본 설정으로 초기화")
    config_parser.set_defaults(func=cmd_config)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    args.func(args)


if __name__ == "__main__":
    main()
