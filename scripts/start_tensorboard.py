"""
Скрипт для запуска TensorBoard
"""
import os
import subprocess
import sys
from pathlib import Path
import argparse


def find_latest_log_dir():
    """Находит последнюю директорию с логами"""
    logs_dir = Path('workspace/models/logs')
    if not logs_dir.exists():
        return None
    
    log_dirs = [d for d in logs_dir.iterdir() if d.is_dir()]
    if not log_dirs:
        return None
    
    # Сортируем по времени модификации (последний - самый новый)
    log_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return log_dirs[0]


def main():
    parser = argparse.ArgumentParser(description='Запуск TensorBoard')
    parser.add_argument('--logdir', type=str, default=None,
                       help='Путь к директории с логами (по умолчанию: последняя в workspace/models/logs/)')
    parser.add_argument('--port', type=int, default=6006,
                       help='Порт для TensorBoard (по умолчанию: 6006)')
    parser.add_argument('--host', type=str, default='localhost',
                       help='Хост для TensorBoard (по умолчанию: localhost)')
    args = parser.parse_args()
    
    # Определяем директорию с логами
    if args.logdir:
        log_dir = Path(args.logdir)
    else:
        log_dir = find_latest_log_dir()
        if log_dir is None:
            print("❌ Не найдена директория с логами!")
            print("   Убедитесь, что обучение было запущено и логи сохранены в workspace/models/logs/")
            return
    
    if not log_dir.exists():
        print(f"❌ Директория {log_dir} не существует!")
        return
    
    print("=" * 60)
    print("Запуск TensorBoard")
    print("=" * 60)
    print(f"Логи: {log_dir}")
    print(f"URL: http://{args.host}:{args.port}")
    print("=" * 60)
    
    # Команда для запуска TensorBoard
    cmd = [
        sys.executable, '-m', 'tensorboard.main',
        '--logdir', str(log_dir),
        '--port', str(args.port),
        '--host', args.host
    ]
    
    print(f"\nКоманда: {' '.join(cmd)}")
    print("\nДля доступа через SSH туннель:")
    print(f"  ssh -L {args.port}:localhost:{args.port} user@remote_host")
    print("\nНажмите Ctrl+C для остановки TensorBoard\n")
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n\nTensorBoard остановлен")
    except FileNotFoundError:
        print("❌ TensorBoard не найден!")
        print("   Установите: pip install tensorboard")


if __name__ == '__main__':
    main()

