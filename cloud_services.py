"""
Утилиты для работы с облачными сервисами (Paperspace и Hugging Face):
- Загрузка данных для обучения и скачивание результатов
- Работа с тиковыми данными
- Загрузка и скачивание результатов анализа фичей (--analyze-features)
"""
import os
import tarfile
import argparse
from pathlib import Path
from datetime import datetime
import subprocess
import sys
from typing import Optional, List
import shutil

try:
    from huggingface_hub import HfApi, upload_folder, snapshot_download
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("⚠️  huggingface_hub не установлен. Установите: pip install huggingface_hub")


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
    
    def create_paperspace_training_archive(self, 
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
    
    def upload_paperspace_training_data(self, archive_path: str, method: str = 'scp') -> bool:
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
    
    # Алиасы для обратной совместимости
    create_training_archive = create_paperspace_training_archive
    upload_training_data = upload_paperspace_training_data


class PaperspaceDownloader:
    """Класс для скачивания результатов обучения с Paperspace"""
    
    def __init__(self, host: str = 'paperspace.com', path: str = '/storage/', user: Optional[str] = None):
        self.host = host
        self.path = path
        self.user = user
    
    def create_paperspace_results_archive(self, output_file: str) -> bool:
        """
        Создает tar.gz архив с результатами обучения для Paperspace
        
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
    
    def download_paperspace_results(self, remote_archive: str, local_path: str = '.', method: str = 'scp') -> bool:
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
    
    def list_paperspace_files(self, remote_path: str = None) -> bool:
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
    
    # Алиасы для обратной совместимости
    create_results_archive = create_paperspace_results_archive
    download_results = download_paperspace_results
    list_remote_files = list_paperspace_files


class HuggingFaceUploader:
    """Класс для загрузки данных на Hugging Face Hub"""
    
    def __init__(self, repo_id: str, token: Optional[str] = None):
        """
        Args:
            repo_id: ID репозитория на Hugging Face (например, 'username/dataset-name')
            token: Hugging Face токен (если None, используется из переменной окружения HF_TOKEN)
        """
        if not HF_AVAILABLE:
            raise ImportError("huggingface_hub не установлен. Установите: pip install huggingface_hub")
        
        self.repo_id = repo_id
        self.api = HfApi(token=token)
        self.token = token or os.getenv('HF_TOKEN')
        
        if not self.token:
            print("⚠️  Предупреждение: HF_TOKEN не установлен. Может потребоваться авторизация.")
    
    def upload_hf_ticks(self, ticks_dir: str = 'workspace/raw_data/ticks', 
                     commit_message: str = "Upload tick data") -> bool:
        """
        Загружает тиковые данные на Hugging Face
        
        Args:
            ticks_dir: Директория с тиковыми данными
            commit_message: Сообщение коммита
        """
        print("=" * 60)
        print("Загрузка тиковых данных на Hugging Face")
        print("=" * 60)
        
        ticks_path = Path(ticks_dir)
        if not ticks_path.exists():
            print(f"❌ Директория {ticks_dir} не найдена!")
            return False
        
        size = get_directory_size(ticks_path)
        print(f"📊 Размер тиковых данных: {format_size(size)}")
        print(f"📁 Репозиторий: {self.repo_id}")
        print(f"📂 Директория: {ticks_dir}")
        
        try:
            # Создаем временную директорию для загрузки
            temp_dir = Path('temp_hf_upload')
            temp_ticks_dir = temp_dir / 'ticks'
            temp_ticks_dir.mkdir(parents=True, exist_ok=True)
            
            # Копируем тики во временную директорию
            print(f"\nКопирование данных...")
            shutil.copytree(ticks_path, temp_ticks_dir, dirs_exist_ok=True)
            
            # Загружаем на Hugging Face
            print(f"\nЗагрузка на Hugging Face...")
            upload_folder(
                folder_path=str(temp_dir),
                repo_id=self.repo_id,
                repo_type="dataset",
                token=self.token,
                commit_message=commit_message
            )
            
            # Удаляем временную директорию
            shutil.rmtree(temp_dir, ignore_errors=True)
            
            print(f"\n✓ Тиковые данные успешно загружены на Hugging Face!")
            print(f"  Репозиторий: https://huggingface.co/datasets/{self.repo_id}")
            return True
            
        except Exception as e:
            print(f"❌ Ошибка при загрузке: {e}")
            shutil.rmtree(temp_dir, ignore_errors=True)
            return False
    
    def upload_hf_feature_analysis(self, 
                               analysis_dir: str = 'workspace/features-analysis',
                               commit_message: str = "Upload feature analysis results") -> bool:
        """
        Загружает результаты анализа фичей (--analyze-features) на Hugging Face
        
        Args:
            analysis_dir: Директория с результатами анализа (по умолчанию: workspace/features-analysis)
            commit_message: Сообщение коммита
        """
        print("=" * 60)
        print("Загрузка результатов анализа фичей на Hugging Face")
        print("=" * 60)
        
        analysis_path = Path(analysis_dir)
        if not analysis_path.exists():
            print(f"❌ Директория {analysis_dir} не найдена!")
            print("   Сначала запустите: python full_pipeline.py --analyze-features")
            return False
        
        # Проверяем наличие основных файлов
        expected_files = [
            'feature_statistics.csv',
            'feature_importance.csv',
            'outliers_analysis.csv',
            'feature_by_class_statistics.csv',
            'feature_analysis_report.html'
        ]
        
        found_files = []
        for file_name in expected_files:
            file_path = analysis_path / file_name
            if file_path.exists():
                found_files.append(file_name)
                size = file_path.stat().st_size
                print(f"✓ Найден: {file_name} ({format_size(size)})")
            else:
                print(f"⚠ Отсутствует: {file_name}")
        
        if not found_files:
            print("❌ Не найдено ни одного файла результатов анализа!")
            return False
        
        # Проверяем наличие директории plots (опционально)
        plots_dir = analysis_path / 'plots'
        has_plots = plots_dir.exists() and plots_dir.is_dir()
        if has_plots:
            plots_size = get_directory_size(plots_dir)
            print(f"✓ Найдена директория plots ({format_size(plots_size)})")
        
        size = get_directory_size(analysis_path)
        print(f"\n📊 Общий размер результатов: {format_size(size)}")
        print(f"📁 Репозиторий: {self.repo_id}")
        print(f"📂 Директория: {analysis_dir}")
        
        try:
            # Создаем временную директорию для загрузки
            temp_dir = Path('temp_hf_upload')
            temp_analysis_dir = temp_dir / 'features-analysis'
            temp_analysis_dir.mkdir(parents=True, exist_ok=True)
            
            # Копируем все файлы и директории
            print(f"\nПодготовка данных...")
            for item in analysis_path.iterdir():
                if item.is_file():
                    shutil.copy2(item, temp_analysis_dir / item.name)
                    print(f"  Скопирован файл: {item.name}")
                elif item.is_dir():
                    shutil.copytree(item, temp_analysis_dir / item.name, dirs_exist_ok=True)
                    print(f"  Скопирована директория: {item.name}")
            
            # Загружаем на Hugging Face
            print(f"\nЗагрузка на Hugging Face...")
            upload_folder(
                folder_path=str(temp_dir),
                repo_id=self.repo_id,
                repo_type="dataset",
                token=self.token,
                commit_message=commit_message
            )
            
            # Удаляем временную директорию
            shutil.rmtree(temp_dir, ignore_errors=True)
            
            print(f"\n✓ Результаты анализа фичей успешно загружены на Hugging Face!")
            print(f"  Репозиторий: https://huggingface.co/datasets/{self.repo_id}")
            return True
            
        except Exception as e:
            print(f"❌ Ошибка при загрузке: {e}")
            import traceback
            traceback.print_exc()
            shutil.rmtree(temp_dir, ignore_errors=True)
            return False
    
    def upload_hf_training_data(self,
                             include_scalers: bool = True,
                             include_cache: bool = False,
                             commit_message: str = "Upload training data") -> bool:
        """
        Загружает данные для обучения на Hugging Face (без тиков)
        
        Args:
            include_scalers: Включать ли scalers
            include_cache: Включать ли кэши
            commit_message: Сообщение коммита
        """
        print("=" * 60)
        print("Загрузка данных для обучения на Hugging Face")
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
        
        # Scalers (опционально)
        if include_scalers:
            scalers_dir = Path('workspace/prepared/scalers')
            if scalers_dir.exists():
                size = get_directory_size(scalers_dir)
                paths_to_include.append(str(scalers_dir))
                print(f"✓ Включена директория: {scalers_dir} ({format_size(size)})")
        
        # Кэши (опционально)
        if include_cache:
            cache_dir = Path('workspace/raw_data/cache')
            if cache_dir.exists():
                size = get_directory_size(cache_dir)
                paths_to_include.append(str(cache_dir))
                print(f"✓ Включена директория: {cache_dir} ({format_size(size)})")
        
        if not paths_to_include:
            print("❌ Нет данных для загрузки!")
            return False
        
        print(f"\n📁 Репозиторий: {self.repo_id}")
        
        try:
            # Создаем временную директорию для загрузки
            temp_dir = Path('temp_hf_upload')
            temp_dir.mkdir(parents=True, exist_ok=True)
            
            # Копируем файлы во временную директорию с сохранением структуры
            print(f"\nПодготовка данных...")
            for path_str in paths_to_include:
                path = Path(path_str)
                if path.exists():
                    # Сохраняем структуру директорий
                    rel_path = path.relative_to(Path('workspace').parent)
                    dest_path = temp_dir / rel_path
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    if path.is_file():
                        shutil.copy2(path, dest_path)
                    else:
                        shutil.copytree(path, dest_path, dirs_exist_ok=True)
                    print(f"  Скопировано: {path_str}")
            
            # Загружаем на Hugging Face
            print(f"\nЗагрузка на Hugging Face...")
            upload_folder(
                folder_path=str(temp_dir),
                repo_id=self.repo_id,
                repo_type="dataset",
                token=self.token,
                commit_message=commit_message
            )
            
            # Удаляем временную директорию
            shutil.rmtree(temp_dir, ignore_errors=True)
            
            print(f"\n✓ Данные для обучения успешно загружены на Hugging Face!")
            print(f"  Репозиторий: https://huggingface.co/datasets/{self.repo_id}")
            return True
            
        except Exception as e:
            print(f"❌ Ошибка при загрузке: {e}")
            import traceback
            traceback.print_exc()
            shutil.rmtree(temp_dir, ignore_errors=True)
            return False
    
    # Алиасы для обратной совместимости
    upload_ticks = upload_hf_ticks
    upload_feature_analysis = upload_hf_feature_analysis
    upload_training_data = upload_hf_training_data


class HuggingFaceDownloader:
    """Класс для скачивания данных с Hugging Face Hub"""
    
    def __init__(self, repo_id: str, token: Optional[str] = None):
        """
        Args:
            repo_id: ID репозитория на Hugging Face (например, 'username/dataset-name')
            token: Hugging Face токен (если None, используется из переменной окружения HF_TOKEN)
        """
        if not HF_AVAILABLE:
            raise ImportError("huggingface_hub не установлен. Установите: pip install huggingface_hub")
        
        self.repo_id = repo_id
        self.api = HfApi(token=token)
        self.token = token or os.getenv('HF_TOKEN')
    
    def download_hf_ticks(self, local_dir: str = 'workspace/raw_data/ticks') -> bool:
        """
        Скачивает тиковые данные с Hugging Face
        
        Args:
            local_dir: Локальная директория для сохранения
        """
        print("=" * 60)
        print("Скачивание тиковых данных с Hugging Face")
        print("=" * 60)
        
        print(f"📁 Репозиторий: {self.repo_id}")
        print(f"📂 Локальная директория: {local_dir}")
        
        try:
            local_path = Path(local_dir)
            local_path.mkdir(parents=True, exist_ok=True)
            
            # Скачиваем данные
            print(f"\nСкачивание данных...")
            downloaded_path = snapshot_download(
                repo_id=self.repo_id,
                repo_type="dataset",
                local_dir=str(local_path),
                token=self.token
            )
            
            # Если данные скачались в поддиректорию ticks, перемещаем их
            downloaded_path = Path(downloaded_path)
            ticks_subdir = downloaded_path / 'ticks'
            if ticks_subdir.exists() and ticks_subdir.is_dir():
                # Данные в поддиректории ticks, перемещаем содержимое (не копируем!)
                print(f"  Перемещение данных из поддиректории...")
                for item in ticks_subdir.iterdir():
                    dest = local_path / item.name
                    if item.is_file():
                        if dest.exists():
                            dest.unlink()  # Удаляем существующий файл
                        shutil.move(str(item), str(dest))  # Перемещаем
                    else:
                        if dest.exists():
                            shutil.rmtree(dest)
                        shutil.move(str(item), str(dest))  # Перемещаем директорию
                
                # Удаляем пустую поддиректорию ticks после перемещения
                try:
                    ticks_subdir.rmdir()  # Удаляем пустую директорию
                except OSError:
                    # Если директория не пустая, удаляем рекурсивно
                    shutil.rmtree(ticks_subdir)
            
            print(f"\n✓ Тиковые данные успешно скачаны!")
            print(f"  Локальная директория: {local_dir}")
            return True
            
        except Exception as e:
            print(f"❌ Ошибка при скачивании: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def download_hf_training_data(self, local_dir: str = 'workspace') -> bool:
        """
        Скачивает данные для обучения с Hugging Face
        
        Args:
            local_dir: Локальная директория для сохранения
        """
        print("=" * 60)
        print("Скачивание данных для обучения с Hugging Face")
        print("=" * 60)
        
        print(f"📁 Репозиторий: {self.repo_id}")
        print(f"📂 Локальная директория: {local_dir}")
        
        try:
            local_path = Path(local_dir)
            local_path.mkdir(parents=True, exist_ok=True)
            
            # Скачиваем данные
            print(f"\nСкачивание данных...")
            snapshot_download(
                repo_id=self.repo_id,
                repo_type="dataset",
                local_dir=str(local_path),
                token=self.token
            )
            
            print(f"\n✓ Данные для обучения успешно скачаны!")
            print(f"  Локальная директория: {local_dir}")
            return True
            
        except Exception as e:
            print(f"❌ Ошибка при скачивании: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def download_hf_feature_analysis(self, local_dir: str = 'workspace/features-analysis') -> bool:
        """
        Скачивает результаты анализа фичей с Hugging Face
        
        Args:
            local_dir: Локальная директория для сохранения (по умолчанию: workspace/features-analysis)
        """
        print("=" * 60)
        print("Скачивание результатов анализа фичей с Hugging Face")
        print("=" * 60)
        
        print(f"📁 Репозиторий: {self.repo_id}")
        print(f"📂 Локальная директория: {local_dir}")
        
        try:
            local_path = Path(local_dir)
            local_path.mkdir(parents=True, exist_ok=True)
            
            # Скачиваем данные
            print(f"\nСкачивание данных...")
            downloaded_path = snapshot_download(
                repo_id=self.repo_id,
                repo_type="dataset",
                local_dir=str(local_path.parent),
                token=self.token
            )
            
            # Если данные скачались в поддиректорию features-analysis, перемещаем их
            downloaded_path = Path(downloaded_path)
            analysis_subdir = downloaded_path / 'features-analysis'
            if analysis_subdir.exists() and analysis_subdir.is_dir():
                # Данные в поддиректории features-analysis, перемещаем содержимое
                print(f"  Перемещение данных из поддиректории...")
                for item in analysis_subdir.iterdir():
                    dest = local_path / item.name
                    if item.is_file():
                        if dest.exists():
                            dest.unlink()  # Удаляем существующий файл
                        shutil.move(str(item), str(dest))  # Перемещаем
                    else:
                        if dest.exists():
                            shutil.rmtree(dest)
                        shutil.move(str(item), str(dest))  # Перемещаем директорию
                
                # Удаляем пустую поддиректорию features-analysis после перемещения
                try:
                    analysis_subdir.rmdir()  # Удаляем пустую директорию
                except OSError:
                    # Если директория не пустая, удаляем рекурсивно
                    shutil.rmtree(analysis_subdir)
            
            print(f"\n✓ Результаты анализа фичей успешно скачаны!")
            print(f"  Локальная директория: {local_dir}")
            print(f"\n  Доступные файлы:")
            for item in local_path.iterdir():
                if item.is_file():
                    size = item.stat().st_size
                    print(f"    - {item.name} ({format_size(size)})")
                elif item.is_dir():
                    size = get_directory_size(item)
                    print(f"    - {item.name}/ ({format_size(size)})")
            return True
            
        except Exception as e:
            print(f"❌ Ошибка при скачивании: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # Алиасы для обратной совместимости
    download_ticks = download_hf_ticks
    download_training_data = download_hf_training_data
    download_feature_analysis = download_hf_feature_analysis


def main():
    parser = argparse.ArgumentParser(
        description='Утилиты для работы с облачными сервисами (Paperspace и Hugging Face)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:

Paperspace:
  python cloud_services.py upload-training --host paperspace.com --path /storage/
  python cloud_services.py create-training-archive --output training_data.tar.gz
  python cloud_services.py download-results --host paperspace.com --path /storage/results.tar.gz
  python cloud_services.py create-results-archive --output results.tar.gz
  python cloud_services.py list-remote-files --host paperspace.com --path /storage/

Hugging Face:
  python cloud_services.py hf-upload-ticks --repo-id username/dataset-name
  python cloud_services.py hf-download-ticks --repo-id username/dataset-name
  python cloud_services.py hf-upload-training --repo-id username/dataset-name
  python cloud_services.py hf-download-training --repo-id username/dataset-name
  python cloud_services.py hf-upload-features --repo-id username/dataset-name
  python cloud_services.py hf-download-features --repo-id username/dataset-name
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
    
    # Hugging Face: Upload ticks
    hf_upload_ticks_parser = subparsers.add_parser('hf-upload-ticks', help='Загрузить тики на Hugging Face')
    hf_upload_ticks_parser.add_argument('--repo-id', type=str, required=True, help='ID репозитория (username/dataset-name)')
    hf_upload_ticks_parser.add_argument('--token', type=str, default=None, help='Hugging Face токен (или HF_TOKEN env var)')
    hf_upload_ticks_parser.add_argument('--ticks-dir', type=str, default='workspace/raw_data/ticks', help='Директория с тиками')
    hf_upload_ticks_parser.add_argument('--commit-message', type=str, default='Upload tick data', help='Сообщение коммита')
    
    # Hugging Face: Download ticks
    hf_download_ticks_parser = subparsers.add_parser('hf-download-ticks', help='Скачать тики с Hugging Face')
    hf_download_ticks_parser.add_argument('--repo-id', type=str, required=True, help='ID репозитория (username/dataset-name)')
    hf_download_ticks_parser.add_argument('--token', type=str, default=None, help='Hugging Face токен (или HF_TOKEN env var)')
    hf_download_ticks_parser.add_argument('--local-dir', type=str, default='workspace/raw_data/ticks', help='Локальная директория')
    
    # Hugging Face: Upload training data
    hf_upload_training_parser = subparsers.add_parser('hf-upload-training', help='Загрузить данные для обучения на Hugging Face')
    hf_upload_training_parser.add_argument('--repo-id', type=str, required=True, help='ID репозитория (username/dataset-name)')
    hf_upload_training_parser.add_argument('--token', type=str, default=None, help='Hugging Face токен (или HF_TOKEN env var)')
    hf_upload_training_parser.add_argument('--include-scalers', action='store_true', default=True, help='Включить scalers')
    hf_upload_training_parser.add_argument('--no-scalers', action='store_false', dest='include_scalers', help='Не включать scalers')
    hf_upload_training_parser.add_argument('--include-cache', action='store_true', help='Включить кэши')
    hf_upload_training_parser.add_argument('--commit-message', type=str, default='Upload training data', help='Сообщение коммита')
    
    # Hugging Face: Download training data
    hf_download_training_parser = subparsers.add_parser('hf-download-training', help='Скачать данные для обучения с Hugging Face')
    hf_download_training_parser.add_argument('--repo-id', type=str, required=True, help='ID репозитория (username/dataset-name)')
    hf_download_training_parser.add_argument('--token', type=str, default=None, help='Hugging Face токен (или HF_TOKEN env var)')
    hf_download_training_parser.add_argument('--local-dir', type=str, default='workspace', help='Локальная директория')
    
    # Hugging Face: Upload feature analysis
    hf_upload_features_parser = subparsers.add_parser('hf-upload-features', help='Загрузить результаты анализа фичей на Hugging Face')
    hf_upload_features_parser.add_argument('--repo-id', type=str, required=True, help='ID репозитория (username/dataset-name)')
    hf_upload_features_parser.add_argument('--token', type=str, default=None, help='Hugging Face токен (или HF_TOKEN env var)')
    hf_upload_features_parser.add_argument('--analysis-dir', type=str, default='workspace/features-analysis', help='Директория с результатами анализа')
    hf_upload_features_parser.add_argument('--commit-message', type=str, default='Upload feature analysis results', help='Сообщение коммита')
    
    # Hugging Face: Download feature analysis
    hf_download_features_parser = subparsers.add_parser('hf-download-features', help='Скачать результаты анализа фичей с Hugging Face')
    hf_download_features_parser.add_argument('--repo-id', type=str, required=True, help='ID репозитория (username/dataset-name)')
    hf_download_features_parser.add_argument('--token', type=str, default=None, help='Hugging Face токен (или HF_TOKEN env var)')
    hf_download_features_parser.add_argument('--local-dir', type=str, default='workspace/features-analysis', help='Локальная директория')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    if args.command == 'upload-training':
        uploader = PaperspaceUploader(host=args.host, path=args.path, user=args.user)
        archive_name = f'training_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.tar.gz'
        if uploader.create_paperspace_training_archive(archive_name,
                                           include_ticks=args.include_ticks,
                                           include_cache=args.include_cache,
                                           ask_ticks=not args.no_ask_ticks):
            uploader.upload_paperspace_training_data(archive_name, method=args.method)
    
    elif args.command == 'create-training-archive':
        uploader = PaperspaceUploader()
        uploader.create_paperspace_training_archive(
            output_file=args.output,
            include_ticks=args.include_ticks,
            include_cache=args.include_cache,
            ask_ticks=not args.no_ask_ticks
        )
    
    elif args.command == 'download-results':
        downloader = PaperspaceDownloader(host=args.host, user=args.user)
        downloader.download_paperspace_results(args.path, local_path=args.local_path, method=args.method)
    
    elif args.command == 'create-results-archive':
        downloader = PaperspaceDownloader()
        downloader.create_paperspace_results_archive(args.output)
    
    elif args.command == 'list-remote-files':
        downloader = PaperspaceDownloader(host=args.host, path=args.path, user=args.user)
        downloader.list_paperspace_files()
    
    elif args.command == 'hf-upload-ticks':
        if not HF_AVAILABLE:
            print("❌ huggingface_hub не установлен. Установите: pip install huggingface_hub")
            return
        uploader = HuggingFaceUploader(repo_id=args.repo_id, token=args.token)
        uploader.upload_hf_ticks(ticks_dir=args.ticks_dir, commit_message=args.commit_message)
    
    elif args.command == 'hf-download-ticks':
        if not HF_AVAILABLE:
            print("❌ huggingface_hub не установлен. Установите: pip install huggingface_hub")
            return
        downloader = HuggingFaceDownloader(repo_id=args.repo_id, token=args.token)
        downloader.download_hf_ticks(local_dir=args.local_dir)
    
    elif args.command == 'hf-upload-training':
        if not HF_AVAILABLE:
            print("❌ huggingface_hub не установлен. Установите: pip install huggingface_hub")
            return
        uploader = HuggingFaceUploader(repo_id=args.repo_id, token=args.token)
        uploader.upload_hf_training_data(
            include_scalers=args.include_scalers,
            include_cache=args.include_cache,
            commit_message=args.commit_message
        )
    
    elif args.command == 'hf-download-training':
        if not HF_AVAILABLE:
            print("❌ huggingface_hub не установлен. Установите: pip install huggingface_hub")
            return
        downloader = HuggingFaceDownloader(repo_id=args.repo_id, token=args.token)
        downloader.download_hf_training_data(local_dir=args.local_dir)
    
    elif args.command == 'hf-upload-features':
        if not HF_AVAILABLE:
            print("❌ huggingface_hub не установлен. Установите: pip install huggingface_hub")
            return
        uploader = HuggingFaceUploader(repo_id=args.repo_id, token=args.token)
        uploader.upload_hf_feature_analysis(
            analysis_dir=args.analysis_dir,
            commit_message=args.commit_message
        )
    
    elif args.command == 'hf-download-features':
        if not HF_AVAILABLE:
            print("❌ huggingface_hub не установлен. Установите: pip install huggingface_hub")
            return
        downloader = HuggingFaceDownloader(repo_id=args.repo_id, token=args.token)
        downloader.download_hf_feature_analysis(local_dir=args.local_dir)
    
    print("\n" + "=" * 60)
    print("Готово!")
    print("=" * 60)


if __name__ == '__main__':
    main()

