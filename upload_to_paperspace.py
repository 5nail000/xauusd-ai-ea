"""
Скрипт для упаковки и отправки всех данных на Paperspace
"""
import os
import tarfile
import argparse
from pathlib import Path
from datetime import datetime
import subprocess
import sys

def get_directory_size(path: Path) -> int:
    """Возвращает размер директории в байтах"""
    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if os.path.exists(filepath):
                total += os.path.getsize(filepath)
    return total

def format_size(size_bytes: int) -> str:
    """Форматирует размер в читаемый вид"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"

def create_tarball(output_file: str, 
                  include_ticks: bool = True,
                  include_prepared: bool = True,
                  include_train_data: bool = True,
                  include_models: bool = False,
                  exclude_patterns: list = None):
    """
    Создает tar.gz архив со всеми необходимыми данными
    
    Args:
        output_file: Путь к выходному файлу
        include_ticks: Включать ли тиковые данные
        include_prepared: Включать ли подготовленные данные
        include_train_data: Включать ли train/val/test CSV
        include_models: Включать ли обученные модели
        exclude_patterns: Паттерны файлов для исключения
    """
    exclude_patterns = exclude_patterns or []
    
    print("=" * 60)
    print("Упаковка данных для Paperspace")
    print("=" * 60)
    
    # Определяем что включать
    paths_to_include = []
    
    # CSV файлы для обучения (обязательно)
    if include_train_data:
        csv_files = ['data/gold_train.csv', 'data/gold_val.csv', 'data/gold_test.csv']
        for csv_file in csv_files:
            if Path(csv_file).exists():
                paths_to_include.append(csv_file)
                print(f"✓ Включен: {csv_file} ({format_size(Path(csv_file).stat().st_size)})")
            else:
                print(f"⚠ Предупреждение: {csv_file} не найден")
    
    # Подготовленные данные
    if include_prepared:
        prepared_dir = Path('data/prepared')
        if prepared_dir.exists():
            paths_to_include.append(str(prepared_dir))
            size = get_directory_size(prepared_dir)
            print(f"✓ Включена директория: {prepared_dir} ({format_size(size)})")
    
    # Тиковые данные
    if include_ticks:
        ticks_dir = Path('data/ticks')
        if ticks_dir.exists():
            # Проверяем размер перед включением
            size = get_directory_size(ticks_dir)
            print(f"📊 Размер тиковых данных: {format_size(size)}")
            
            response = input(f"Включить тиковые данные ({format_size(size)})? (y/n): ").strip().lower()
            if response in ['y', 'yes', 'да', 'д', '']:
                paths_to_include.append(str(ticks_dir))
                print(f"✓ Включена директория: {ticks_dir}")
            else:
                print("✗ Тиковые данные исключены")
    
    # Модели (опционально)
    if include_models:
        models_dir = Path('models')
        if models_dir.exists():
            paths_to_include.append(str(models_dir))
            size = get_directory_size(models_dir)
            print(f"✓ Включена директория: {models_dir} ({format_size(size)})")
    
    if not paths_to_include:
        print("❌ Нет данных для упаковки!")
        return False
    
    # Создаем архив
    print("\n" + "=" * 60)
    print("Создание архива...")
    print("=" * 60)
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with tarfile.open(output_file, 'w:gz') as tar:
        for path_str in paths_to_include:
            path = Path(path_str)
            if path.exists():
                print(f"Добавление: {path_str}...")
                tar.add(path_str, arcname=path_str, recursive=True)
            else:
                print(f"⚠ Предупреждение: {path_str} не существует, пропускаем")
    
    archive_size = Path(output_file).stat().st_size
    print(f"\n✓ Архив создан: {output_file}")
    print(f"  Размер: {format_size(archive_size)}")
    
    return True

def upload_via_scp(archive_path: str, paperspace_host: str, paperspace_path: str, 
                   paperspace_user: str = None):
    """
    Загружает архив на Paperspace через scp
    
    Args:
        archive_path: Путь к архиву
        paperspace_host: Хост Paperspace (например, paperspace.com или IP)
        paperspace_path: Путь на Paperspace (например, /storage/xauusd-ai-ea/)
        paperspace_user: Пользователь (если нужен)
    """
    print("\n" + "=" * 60)
    print("Загрузка на Paperspace через SCP")
    print("=" * 60)
    
    if paperspace_user:
        scp_target = f"{paperspace_user}@{paperspace_host}:{paperspace_path}"
    else:
        scp_target = f"{paperspace_host}:{paperspace_path}"
    
    print(f"Загрузка {archive_path} -> {scp_target}")
    
    cmd = ['scp', archive_path, scp_target]
    print(f"Команда: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
        print("✓ Загрузка завершена успешно!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Ошибка при загрузке: {e}")
        return False
    except FileNotFoundError:
        print("❌ SCP не найден. Установите OpenSSH или используйте другой метод загрузки.")
        print("\nАльтернативные методы:")
        print("1. Загрузите архив через Paperspace UI (Storage -> Upload)")
        print("2. Используйте rsync вместо scp")
        return False

def upload_via_rsync(archive_path: str, paperspace_host: str, paperspace_path: str,
                     paperspace_user: str = None):
    """
    Загружает архив на Paperspace через rsync
    """
    print("\n" + "=" * 60)
    print("Загрузка на Paperspace через RSYNC")
    print("=" * 60)
    
    if paperspace_user:
        rsync_target = f"{paperspace_user}@{paperspace_host}:{paperspace_path}"
    else:
        rsync_target = f"{paperspace_host}:{paperspace_path}"
    
    cmd = ['rsync', '-avz', '--progress', archive_path, rsync_target]
    print(f"Команда: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
        print("✓ Загрузка завершена успешно!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Ошибка при загрузке: {e}")
        return False
    except FileNotFoundError:
        print("❌ RSYNC не найден. Установите rsync или используйте другой метод.")
        return False

def main():
    parser = argparse.ArgumentParser(
        description='Упаковка и отправка данных на Paperspace',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  # Создать архив со всеми данными
  python upload_to_paperspace.py --create-archive

  # Создать архив без тиков (меньше размер)
  python upload_to_paperspace.py --create-archive --no-ticks

  # Создать и загрузить через SCP
  python upload_to_paperspace.py --create-archive --upload-scp --host paperspace.com --path /storage/

  # Только создать архив (загрузить вручную через UI)
  python upload_to_paperspace.py --create-archive --output data_for_paperspace.tar.gz
        """
    )
    
    parser.add_argument('--create-archive', action='store_true',
                       help='Создать tar.gz архив')
    parser.add_argument('--output', '-o', type=str, 
                       default=f'data_for_paperspace_{datetime.now().strftime("%Y%m%d_%H%M%S")}.tar.gz',
                       help='Путь к выходному архиву')
    parser.add_argument('--no-ticks', action='store_true',
                       help='Не включать тиковые данные')
    parser.add_argument('--no-prepared', action='store_true',
                       help='Не включать подготовленные данные')
    parser.add_argument('--no-train-data', action='store_true',
                       help='Не включать train/val/test CSV')
    parser.add_argument('--include-models', action='store_true',
                       help='Включить обученные модели')
    
    # Параметры загрузки
    parser.add_argument('--upload-scp', action='store_true',
                       help='Загрузить через SCP после создания архива')
    parser.add_argument('--upload-rsync', action='store_true',
                       help='Загрузить через RSYNC после создания архива')
    parser.add_argument('--host', type=str, default='paperspace.com',
                       help='Хост Paperspace')
    parser.add_argument('--path', type=str, default='/storage/',
                       help='Путь на Paperspace')
    parser.add_argument('--user', type=str, default=None,
                       help='Пользователь для подключения')
    
    args = parser.parse_args()
    
    if not args.create_archive and not args.upload_scp and not args.upload_rsync:
        parser.print_help()
        return
    
    archive_path = args.output
    
    # Создаем архив
    if args.create_archive:
        success = create_tarball(
            output_file=archive_path,
            include_ticks=not args.no_ticks,
            include_prepared=not args.no_prepared,
            include_train_data=not args.no_train_data,
            include_models=args.include_models
        )
        
        if not success:
            print("❌ Не удалось создать архив")
            return
    
    # Загружаем на Paperspace
    if args.upload_scp:
        if not Path(archive_path).exists():
            print(f"❌ Архив {archive_path} не найден. Сначала создайте его с --create-archive")
            return
        upload_via_scp(archive_path, args.host, args.path, args.user)
    
    elif args.upload_rsync:
        if not Path(archive_path).exists():
            print(f"❌ Архив {archive_path} не найден. Сначала создайте его с --create-archive")
            return
        upload_via_rsync(archive_path, args.host, args.path, args.user)
    
    print("\n" + "=" * 60)
    print("Готово!")
    print("=" * 60)
    print(f"\nСледующие шаги на Paperspace:")
    print(f"1. Распаковать архив: tar -xzf {Path(archive_path).name}")
    print(f"2. Установить зависимости: pip install -r requirements_linux.txt")
    print(f"3. Запустить обучение: python train_model.py")

if __name__ == '__main__':
    main()

