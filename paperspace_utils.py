"""
Утилиты для работы с Paperspace: загрузка данных для обучения и скачивание результатов
"""
import os
import tarfile
import argparse
from pathlib import Path
from datetime import datetime
import subprocess
import sys
from typing import Optional, List


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


class PaperspaceUploader:
    """Класс для загрузки данных на Paperspace для обучения"""
    
    def __init__(self, host: str = 'paperspace.com', path: str = '/storage/', user: Optional[str] = None):
        self.host = host
        self.path = path
        self.user = user
    
    def create_training_archive(self, 
                               output_file: str,
                               include_ticks: bool = False,
                               include_cache: bool = False,
                               ask_ticks: bool = True) -> bool:
        """
        Создает tar.gz архив с данными для обучения
        
        Args:
            output_file: Путь к выходному файлу
            include_ticks: Включать ли тиковые данные
            include_cache: Включать ли кэши
            ask_ticks: Спрашивать ли о включении тиков (если include_ticks=True)
        """
        print("=" * 60)
        print("Упаковка данных для обучения на Paperspace")
        print("=" * 60)
        
        paths_to_include = []
        
        # CSV файлы для обучения (обязательно)
        csv_files = [
            'workspace/prepared/features/gold_train.csv',
            'workspace/prepared/features/gold_val.csv',
            'workspace/prepared/features/gold_test.csv'
        ]
        
        for csv_file in csv_files:
            csv_path = Path(csv_file)
            if csv_path.exists():
                paths_to_include.append(csv_file)
                size = csv_path.stat().st_size
                print(f"✓ Включен: {csv_file} ({format_size(size)})")
            else:
                print(f"⚠ Предупреждение: {csv_file} не найден")
        
        # Кэши (опционально)
        if include_cache:
            cache_dir = Path('workspace/raw_data/cache')
            if cache_dir.exists():
                size = get_directory_size(cache_dir)
                paths_to_include.append(str(cache_dir))
                print(f"✓ Включена директория: {cache_dir} ({format_size(size)})")
        
        # Тиковые данные (опционально, с подтверждением)
        if include_ticks:
            ticks_dir = Path('workspace/raw_data/ticks')
            if ticks_dir.exists():
                size = get_directory_size(ticks_dir)
                print(f"📊 Размер тиковых данных: {format_size(size)}")
                
                if ask_ticks:
                    response = input(f"Включить тиковые данные ({format_size(size)})? (y/n): ").strip().lower()
                    if response not in ['y', 'yes', 'да', 'д', '']:
                        print("✗ Тиковые данные исключены")
                        include_ticks = False
                
                if include_ticks:
                    paths_to_include.append(str(ticks_dir))
                    print(f"✓ Включена директория: {ticks_dir}")
        
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
    
    def upload_training_data(self, archive_path: str, method: str = 'scp') -> bool:
        """
        Загружает архив на Paperspace
        
        Args:
            archive_path: Путь к архиву
            method: Метод загрузки ('scp' или 'rsync')
        """
        if method == 'scp':
            return self._upload_via_scp(archive_path)
        elif method == 'rsync':
            return self._upload_via_rsync(archive_path)
        else:
            print(f"❌ Неизвестный метод: {method}")
            return False
    
    def _upload_via_scp(self, archive_path: str) -> bool:
        """Загружает архив через SCP"""
        print("\n" + "=" * 60)
        print("Загрузка на Paperspace через SCP")
        print("=" * 60)
        
        if self.user:
            scp_target = f"{self.user}@{self.host}:{self.path}"
        else:
            scp_target = f"{self.host}:{self.path}"
        
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
            print("❌ SCP не найден. Установите OpenSSH или используйте rsync.")
            return False
    
    def _upload_via_rsync(self, archive_path: str) -> bool:
        """Загружает архив через RSYNC"""
        print("\n" + "=" * 60)
        print("Загрузка на Paperspace через RSYNC")
        print("=" * 60)
        
        if self.user:
            rsync_target = f"{self.user}@{self.host}:{self.path}"
        else:
            rsync_target = f"{self.host}:{self.path}"
        
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
            print("❌ RSYNC не найден. Установите rsync.")
            return False


class PaperspaceDownloader:
    """Класс для скачивания результатов обучения с Paperspace"""
    
    def __init__(self, host: str = 'paperspace.com', path: str = '/storage/', user: Optional[str] = None):
        self.host = host
        self.path = path
        self.user = user
    
    def create_results_archive(self, output_file: str) -> bool:
        """
        Создает tar.gz архив с результатами обучения
        
        Args:
            output_file: Путь к выходному файлу
        """
        print("=" * 60)
        print("Упаковка результатов обучения")
        print("=" * 60)
        
        paths_to_include = []
        
        # Модели
        models_dir = Path('workspace/models/checkpoints')
        if models_dir.exists():
            size = get_directory_size(models_dir)
            paths_to_include.append(str(models_dir))
            print(f"✓ Включена директория: {models_dir} ({format_size(size)})")
        
        # Метрики
        metrics_dir = Path('workspace/models/metrics')
        if metrics_dir.exists():
            size = get_directory_size(metrics_dir)
            paths_to_include.append(str(metrics_dir))
            print(f"✓ Включена директория: {metrics_dir} ({format_size(size)})")
        
        # Scalers
        scalers_dir = Path('workspace/prepared/scalers')
        if scalers_dir.exists():
            size = get_directory_size(scalers_dir)
            paths_to_include.append(str(scalers_dir))
            print(f"✓ Включена директория: {scalers_dir} ({format_size(size)})")
        
        # TensorBoard логи (опционально)
        logs_dir = Path('workspace/models/logs')
        if logs_dir.exists():
            size = get_directory_size(logs_dir)
            paths_to_include.append(str(logs_dir))
            print(f"✓ Включена директория: {logs_dir} ({format_size(size)})")
        
        if not paths_to_include:
            print("❌ Нет результатов для упаковки!")
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
        
        archive_size = Path(output_file).stat().st_size
        print(f"\n✓ Архив создан: {output_file}")
        print(f"  Размер: {format_size(archive_size)}")
        
        return True
    
    def download_results(self, remote_archive: str, local_path: str = '.', method: str = 'scp') -> bool:
        """
        Скачивает результаты обучения с Paperspace
        
        Args:
            remote_archive: Путь к архиву на Paperspace
            local_path: Локальная директория для сохранения
            method: Метод скачивания ('scp' или 'rsync')
        """
        if method == 'scp':
            return self._download_via_scp(remote_archive, local_path)
        elif method == 'rsync':
            return self._download_via_rsync(remote_archive, local_path)
        else:
            print(f"❌ Неизвестный метод: {method}")
            return False
    
    def _download_via_scp(self, remote_archive: str, local_path: str) -> bool:
        """Скачивает архив через SCP"""
        print("\n" + "=" * 60)
        print("Скачивание результатов через SCP")
        print("=" * 60)
        
        if self.user:
            scp_source = f"{self.user}@{self.host}:{remote_archive}"
        else:
            scp_source = f"{self.host}:{remote_archive}"
        
        local_file = Path(local_path) / Path(remote_archive).name
        print(f"Скачивание {scp_source} -> {local_file}")
        
        cmd = ['scp', scp_source, str(local_file)]
        print(f"Команда: {' '.join(cmd)}")
        
        try:
            subprocess.run(cmd, check=True)
            print("✓ Скачивание завершено успешно!")
            print(f"  Распакуйте архив: tar -xzf {local_file.name}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Ошибка при скачивании: {e}")
            return False
        except FileNotFoundError:
            print("❌ SCP не найден. Установите OpenSSH.")
            return False
    
    def _download_via_rsync(self, remote_archive: str, local_path: str) -> bool:
        """Скачивает архив через RSYNC"""
        print("\n" + "=" * 60)
        print("Скачивание результатов через RSYNC")
        print("=" * 60)
        
        if self.user:
            rsync_source = f"{self.user}@{self.host}:{remote_archive}"
        else:
            rsync_source = f"{self.host}:{remote_archive}"
        
        local_file = Path(local_path) / Path(remote_archive).name
        cmd = ['rsync', '-avz', '--progress', rsync_source, str(local_file)]
        print(f"Команда: {' '.join(cmd)}")
        
        try:
            subprocess.run(cmd, check=True)
            print("✓ Скачивание завершено успешно!")
            print(f"  Распакуйте архив: tar -xzf {local_file.name}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Ошибка при скачивании: {e}")
            return False
        except FileNotFoundError:
            print("❌ RSYNC не найден. Установите rsync.")
            return False
    
    def list_remote_files(self, remote_path: str = None) -> bool:
        """
        Выводит список файлов на Paperspace
        
        Args:
            remote_path: Путь на удаленном сервере (по умолчанию self.path)
        """
        if remote_path is None:
            remote_path = self.path
        
        print("\n" + "=" * 60)
        print("Список файлов на Paperspace")
        print("=" * 60)
        
        if self.user:
            ssh_target = f"{self.user}@{self.host}"
        else:
            ssh_target = self.host
        
        cmd = ['ssh', ssh_target, f'ls -lh {remote_path}']
        print(f"Команда: {' '.join(cmd)}")
        
        try:
            subprocess.run(cmd, check=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Ошибка при получении списка файлов: {e}")
            return False
        except FileNotFoundError:
            print("❌ SSH не найден. Установите OpenSSH.")
            return False


def main():
    parser = argparse.ArgumentParser(
        description='Утилиты для работы с Paperspace',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:

Загрузка данных для обучения:
  python paperspace_utils.py upload-training --host paperspace.com --path /storage/
  python paperspace_utils.py create-training-archive --output training_data.tar.gz

Скачивание результатов:
  python paperspace_utils.py download-results --host paperspace.com --path /storage/results.tar.gz
  python paperspace_utils.py create-results-archive --output results.tar.gz
  python paperspace_utils.py list-remote-files --host paperspace.com --path /storage/
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Команда')
    
    # Upload training data
    upload_parser = subparsers.add_parser('upload-training', help='Загрузить данные для обучения')
    upload_parser.add_argument('--host', type=str, default='paperspace.com', help='Хост Paperspace')
    upload_parser.add_argument('--path', type=str, default='/storage/', help='Путь на Paperspace')
    upload_parser.add_argument('--user', type=str, default=None, help='Пользователь')
    upload_parser.add_argument('--method', type=str, choices=['scp', 'rsync'], default='scp', help='Метод загрузки')
    upload_parser.add_argument('--include-ticks', action='store_true', help='Включить тиковые данные')
    upload_parser.add_argument('--include-cache', action='store_true', help='Включить кэши')
    upload_parser.add_argument('--no-ask-ticks', action='store_true', help='Не спрашивать о тиках')
    
    # Create training archive
    create_training_parser = subparsers.add_parser('create-training-archive', help='Создать архив для обучения')
    create_training_parser.add_argument('--output', '-o', type=str,
                                      default=f'training_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.tar.gz',
                                      help='Путь к выходному архиву')
    create_training_parser.add_argument('--include-ticks', action='store_true', help='Включить тиковые данные')
    create_training_parser.add_argument('--include-cache', action='store_true', help='Включить кэши')
    create_training_parser.add_argument('--no-ask-ticks', action='store_true', help='Не спрашивать о тиках')
    
    # Download results
    download_parser = subparsers.add_parser('download-results', help='Скачать результаты обучения')
    download_parser.add_argument('--host', type=str, default='paperspace.com', help='Хост Paperspace')
    download_parser.add_argument('--path', type=str, required=True, help='Путь к архиву на Paperspace')
    download_parser.add_argument('--user', type=str, default=None, help='Пользователь')
    download_parser.add_argument('--method', type=str, choices=['scp', 'rsync'], default='scp', help='Метод скачивания')
    download_parser.add_argument('--local-path', type=str, default='.', help='Локальная директория')
    
    # Create results archive
    create_results_parser = subparsers.add_parser('create-results-archive', help='Создать архив результатов')
    create_results_parser.add_argument('--output', '-o', type=str,
                                     default=f'results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.tar.gz',
                                     help='Путь к выходному архиву')
    
    # List remote files
    list_parser = subparsers.add_parser('list-remote-files', help='Список файлов на Paperspace')
    list_parser.add_argument('--host', type=str, default='paperspace.com', help='Хост Paperspace')
    list_parser.add_argument('--path', type=str, default='/storage/', help='Путь на Paperspace')
    list_parser.add_argument('--user', type=str, default=None, help='Пользователь')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    if args.command == 'upload-training':
        uploader = PaperspaceUploader(host=args.host, path=args.path, user=args.user)
        archive_name = f'training_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.tar.gz'
        if uploader.create_training_archive(archive_name, 
                                           include_ticks=args.include_ticks,
                                           include_cache=args.include_cache,
                                           ask_ticks=not args.no_ask_ticks):
            uploader.upload_training_data(archive_name, method=args.method)
    
    elif args.command == 'create-training-archive':
        uploader = PaperspaceUploader()
        uploader.create_training_archive(
            output_file=args.output,
            include_ticks=args.include_ticks,
            include_cache=args.include_cache,
            ask_ticks=not args.no_ask_ticks
        )
    
    elif args.command == 'download-results':
        downloader = PaperspaceDownloader(host=args.host, user=args.user)
        downloader.download_results(args.path, local_path=args.local_path, method=args.method)
    
    elif args.command == 'create-results-archive':
        downloader = PaperspaceDownloader()
        downloader.create_results_archive(args.output)
    
    elif args.command == 'list-remote-files':
        downloader = PaperspaceDownloader(host=args.host, path=args.path, user=args.user)
        downloader.list_remote_files()
    
    print("\n" + "=" * 60)
    print("Готово!")
    print("=" * 60)


if __name__ == '__main__':
    main()

