"""
Утилиты для работы с облачными сервисами (Hugging Face):
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
from typing import Optional, List, Dict, Any
import shutil
import time
import json

try:
    from huggingface_hub import HfApi, upload_folder, snapshot_download
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("⚠️  huggingface_hub не установлен. Установите: pip install huggingface_hub")

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("⚠️  requests не установлен. Установите: pip install requests")

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
                               analysis_dir: str = 'workspace/analysis-of-features',
                               commit_message: str = "Upload feature analysis results") -> bool:
        """
        Загружает результаты анализа фичей (--analyze-features) на Hugging Face
        
        Args:
            analysis_dir: Директория с результатами анализа (по умолчанию: workspace/analysis-of-features)
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
        
        # Проверяем наличие excluded_features.txt в workspace
        excluded_features_file = Path('workspace/excluded_features.txt')
        has_excluded = excluded_features_file.exists()
        if has_excluded:
            size = excluded_features_file.stat().st_size
            print(f"✓ Найден: excluded_features.txt ({format_size(size)})")
        
        size = get_directory_size(analysis_path)
        print(f"\n📊 Общий размер результатов: {format_size(size)}")
        print(f"📁 Репозиторий: {self.repo_id}")
        print(f"📂 Директория: {analysis_dir}")
        
        try:
            # Создаем временную директорию для загрузки
            temp_dir = Path('temp_hf_upload')
            temp_analysis_dir = temp_dir / 'analysis-of-features'
            temp_analysis_dir.mkdir(parents=True, exist_ok=True)
            
            # Копируем все файлы и директории из analysis_dir
            print(f"\nПодготовка данных...")
            for item in analysis_path.iterdir():
                if item.is_file():
                    shutil.copy2(item, temp_analysis_dir / item.name)
                    print(f"  Скопирован файл: {item.name}")
                elif item.is_dir():
                    shutil.copytree(item, temp_analysis_dir / item.name, dirs_exist_ok=True)
                    print(f"  Скопирована директория: {item.name}")
            
            # Копируем excluded_features.txt из workspace, если он существует
            if has_excluded:
                temp_workspace_dir = temp_dir / 'workspace'
                temp_workspace_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(excluded_features_file, temp_workspace_dir / 'excluded_features.txt')
                print(f"  Скопирован файл: workspace/excluded_features.txt")
            
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
        
        # Проверяем наличие excluded_features.txt в workspace
        excluded_features_file = Path('workspace/excluded_features.txt')
        has_excluded = excluded_features_file.exists()
        if has_excluded:
            size = excluded_features_file.stat().st_size
            print(f"✓ Найден: excluded_features.txt ({format_size(size)})")
        
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
            
            # Копируем excluded_features.txt, если он существует
            if has_excluded:
                workspace_dest = temp_dir / 'workspace'
                workspace_dest.mkdir(parents=True, exist_ok=True)
                shutil.copy2(excluded_features_file, workspace_dest / 'excluded_features.txt')
                print(f"  Скопировано: workspace/excluded_features.txt")
            
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


class HuggingFaceDeleter:
    """Класс для удаления данных из Hugging Face Hub"""
    
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
    
    def _list_repo_files(self) -> List[str]:
        """Получает список всех файлов в репозитории"""
        try:
            files = self.api.list_repo_files(
                repo_id=self.repo_id,
                repo_type="dataset",
                token=self.token
            )
            return files
        except Exception as e:
            print(f"❌ Ошибка при получении списка файлов: {e}")
            return []
    
    def delete_hf_ticks(self, commit_message: str = "Delete tick data") -> bool:
        """
        Удаляет тиковые данные из репозитория
        
        Args:
            commit_message: Сообщение коммита
        """
        print("=" * 60)
        print("Удаление тиковых данных из Hugging Face")
        print("=" * 60)
        print(f"📁 Репозиторий: {self.repo_id}")
        
        try:
            files = self._list_repo_files()
            tick_files = [f for f in files if f.startswith('ticks/')]
            
            if not tick_files:
                print("✓ Тиковые данные не найдены в репозитории")
                return True
            
            print(f"\nНайдено {len(tick_files)} файлов/директорий для удаления:")
            for file in tick_files[:10]:  # Показываем первые 10
                print(f"  - {file}")
            if len(tick_files) > 10:
                print(f"  ... и еще {len(tick_files) - 10} файлов")
            
            # Удаляем файлы
            print(f"\nУдаление файлов...")
            for file in tick_files:
                try:
                    self.api.delete_file(
                        path_in_repo=file,
                        repo_id=self.repo_id,
                        repo_type="dataset",
                        token=self.token,
                        commit_message=commit_message if file == tick_files[0] else None
                    )
                except Exception as e:
                    print(f"  ⚠️  Ошибка при удалении {file}: {e}")
            
            print(f"\n✓ Тиковые данные успешно удалены из репозитория!")
            return True
            
        except Exception as e:
            print(f"❌ Ошибка при удалении: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def delete_hf_training_data(self, 
                               include_scalers: bool = True,
                               include_cache: bool = True,
                               commit_message: str = "Delete training data") -> bool:
        """
        Удаляет данные для обучения из репозитория
        
        Args:
            include_scalers: Удалять ли scalers
            include_cache: Удалять ли кэши
            commit_message: Сообщение коммита
        """
        print("=" * 60)
        print("Удаление данных для обучения из Hugging Face")
        print("=" * 60)
        print(f"📁 Репозиторий: {self.repo_id}")
        
        try:
            files = self._list_repo_files()
            files_to_delete = []
            
            # CSV файлы для обучения
            training_patterns = [
                'workspace/prepared/features/gold_train.csv',
                'workspace/prepared/features/gold_val.csv',
                'workspace/prepared/features/gold_test.csv'
            ]
            
            # Scalers
            if include_scalers:
                scaler_files = [f for f in files if f.startswith('workspace/prepared/scalers/')]
                files_to_delete.extend(scaler_files)
            
            # Кэши
            if include_cache:
                cache_files = [f for f in files if f.startswith('workspace/raw_data/cache/')]
                files_to_delete.extend(cache_files)
            
            # CSV файлы
            for pattern in training_patterns:
                if pattern in files:
                    files_to_delete.append(pattern)
            
            # excluded_features.txt
            excluded_file = 'workspace/excluded_features.txt'
            if excluded_file in files:
                files_to_delete.append(excluded_file)
            
            if not files_to_delete:
                print("✓ Данные для обучения не найдены в репозитории")
                return True
            
            print(f"\nНайдено {len(files_to_delete)} файлов/директорий для удаления:")
            for file in files_to_delete[:10]:  # Показываем первые 10
                print(f"  - {file}")
            if len(files_to_delete) > 10:
                print(f"  ... и еще {len(files_to_delete) - 10} файлов")
            
            # Удаляем файлы
            print(f"\nУдаление файлов...")
            for i, file in enumerate(files_to_delete):
                try:
                    self.api.delete_file(
                        path_in_repo=file,
                        repo_id=self.repo_id,
                        repo_type="dataset",
                        token=self.token,
                        commit_message=commit_message if i == 0 else None
                    )
                except Exception as e:
                    print(f"  ⚠️  Ошибка при удалении {file}: {e}")
            
            print(f"\n✓ Данные для обучения успешно удалены из репозитория!")
            return True
            
        except Exception as e:
            print(f"❌ Ошибка при удалении: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def delete_hf_feature_analysis(self, commit_message: str = "Delete feature analysis results") -> bool:
        """
        Удаляет результаты анализа фичей из репозитория
        
        Args:
            commit_message: Сообщение коммита
        """
        print("=" * 60)
        print("Удаление результатов анализа фичей из Hugging Face")
        print("=" * 60)
        print(f"📁 Репозиторий: {self.repo_id}")
        
        try:
            files = self._list_repo_files()
            analysis_files = [f for f in files if f.startswith('analysis-of-features/')]
            
            if not analysis_files:
                print("✓ Результаты анализа фичей не найдены в репозитории")
                return True
            
            print(f"\nНайдено {len(analysis_files)} файлов/директорий для удаления:")
            for file in analysis_files[:10]:  # Показываем первые 10
                print(f"  - {file}")
            if len(analysis_files) > 10:
                print(f"  ... и еще {len(analysis_files) - 10} файлов")
            
            # Удаляем файлы
            print(f"\nУдаление файлов...")
            for i, file in enumerate(analysis_files):
                try:
                    self.api.delete_file(
                        path_in_repo=file,
                        repo_id=self.repo_id,
                        repo_type="dataset",
                        token=self.token,
                        commit_message=commit_message if i == 0 else None
                    )
                except Exception as e:
                    print(f"  ⚠️  Ошибка при удалении {file}: {e}")
            
            print(f"\n✓ Результаты анализа фичей успешно удалены из репозитория!")
            return True
            
        except Exception as e:
            print(f"❌ Ошибка при удалении: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def delete_all_data(self, commit_message: str = "Delete all dataset data") -> bool:
        """
        Удаляет все данные из датасета (очищает репозиторий для новых загрузок)
        
        Args:
            commit_message: Сообщение коммита
        """
        print("=" * 60)
        print("Удаление всех данных из датасета Hugging Face")
        print("=" * 60)
        print(f"📁 Репозиторий: {self.repo_id}")
        print("⚠️  ВНИМАНИЕ: Это удалит ВСЕ данные из репозитория!")
        
        try:
            files = self._list_repo_files()
            
            if not files:
                print("✓ Репозиторий уже пуст")
                return True
            
            print(f"\nНайдено {len(files)} файлов/директорий для удаления:")
            for file in files[:20]:  # Показываем первые 20
                print(f"  - {file}")
            if len(files) > 20:
                print(f"  ... и еще {len(files) - 20} файлов")
            
            # Подтверждение
            response = input("\nВы уверены, что хотите удалить ВСЕ данные? (yes/no): ").strip().lower()
            if response != 'yes':
                print("❌ Удаление отменено")
                return False
            
            # Удаляем файлы
            print(f"\nУдаление файлов...")
            for i, file in enumerate(files):
                try:
                    self.api.delete_file(
                        path_in_repo=file,
                        repo_id=self.repo_id,
                        repo_type="dataset",
                        token=self.token,
                        commit_message=commit_message if i == 0 else None
                    )
                    if (i + 1) % 10 == 0:
                        print(f"  Удалено {i + 1}/{len(files)} файлов...")
                except Exception as e:
                    print(f"  ⚠️  Ошибка при удалении {file}: {e}")
            
            print(f"\n✓ Все данные успешно удалены из репозитория!")
            print(f"  Репозиторий готов для новых загрузок")
            return True
            
        except Exception as e:
            print(f"❌ Ошибка при удалении: {e}")
            import traceback
            traceback.print_exc()
            return False


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
            
            # Скачиваем только папку ticks
            print(f"\nСкачивание данных (только папка ticks/)...")
            temp_dir = Path('temp_hf_download')
            temp_dir.mkdir(exist_ok=True)
            
            try:
                downloaded_path = snapshot_download(
                    repo_id=self.repo_id,
                    repo_type="dataset",
                    local_dir=str(temp_dir),
                    token=self.token,
                    allow_patterns=["ticks/**"]  # Скачиваем только папку ticks
                )
                
                # Перемещаем данные из временной директории
                downloaded_path = Path(downloaded_path)
                ticks_source = downloaded_path / 'ticks'
                
                if ticks_source.exists() and ticks_source.is_dir():
                    print(f"  Перемещение данных из временной директории...")
                    for item in ticks_source.iterdir():
                        dest = local_path / item.name
                        if item.is_file():
                            if dest.exists():
                                dest.unlink()
                            shutil.move(str(item), str(dest))
                        else:
                            if dest.exists():
                                shutil.rmtree(dest)
                            shutil.move(str(item), str(dest))
                else:
                    print("⚠️  Папка ticks не найдена в репозитории")
                    print(f"   Проверьте структуру репозитория. Ожидается: ticks/")
                    return False
            finally:
                # Удаляем временную директорию
                if temp_dir.exists():
                    shutil.rmtree(temp_dir, ignore_errors=True)
            
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
            
            # Скачиваем только данные для обучения (CSV файлы, scalers, cache, excluded_features.txt)
            print(f"\nСкачивание данных (только данные для обучения)...")
            temp_dir = Path('temp_hf_download')
            temp_dir.mkdir(exist_ok=True)
            
            try:
                downloaded_path = snapshot_download(
                    repo_id=self.repo_id,
                    repo_type="dataset",
                    local_dir=str(temp_dir),
                    token=self.token,
                    allow_patterns=[
                        "workspace/prepared/features/*.csv",
                        "workspace/prepared/scalers/**",
                        "workspace/raw_data/cache/**",
                        "workspace/excluded_features.txt"  # Скачиваем excluded_features.txt
                    ]  # Скачиваем только нужные данные
                )
                
                # Перемещаем данные из временной директории, сохраняя структуру
                downloaded_path = Path(downloaded_path)
                workspace_source = downloaded_path / 'workspace'
                
                if workspace_source.exists():
                    print(f"  Перемещение данных из временной директории...")
                    found_any = False
                    
                    # Перемещаем prepared/features/*.csv
                    features_source = workspace_source / 'prepared' / 'features'
                    if features_source.exists():
                        features_dest = local_path / 'prepared' / 'features'
                        features_dest.mkdir(parents=True, exist_ok=True)
                        csv_files = list(features_source.glob('*.csv'))
                        if csv_files:
                            found_any = True
                            for csv_file in csv_files:
                                dest_file = features_dest / csv_file.name
                                if dest_file.exists():
                                    dest_file.unlink()
                                shutil.move(str(csv_file), str(dest_file))
                    
                    # Перемещаем prepared/scalers
                    scalers_source = workspace_source / 'prepared' / 'scalers'
                    if scalers_source.exists():
                        found_any = True
                        scalers_dest = local_path / 'prepared' / 'scalers'
                        if scalers_dest.exists():
                            shutil.rmtree(scalers_dest)
                        scalers_dest.parent.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(scalers_source), str(scalers_dest))
                    
                    # Перемещаем raw_data/cache
                    cache_source = workspace_source / 'raw_data' / 'cache'
                    if cache_source.exists():
                        found_any = True
                        cache_dest = local_path / 'raw_data' / 'cache'
                        if cache_dest.exists():
                            shutil.rmtree(cache_dest)
                        cache_dest.parent.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(cache_source), str(cache_dest))
                    
                    # Перемещаем excluded_features.txt, если он найден
                    excluded_source = workspace_source / 'excluded_features.txt'
                    if excluded_source.exists():
                        excluded_dest = local_path / 'excluded_features.txt'
                        if excluded_dest.exists():
                            excluded_dest.unlink()
                        excluded_dest.parent.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(excluded_source), str(excluded_dest))
                        print(f"  ✓ Перемещен excluded_features.txt в workspace/")
                    
                    # Проверяем, что хотя бы что-то было скачано
                    if not found_any:
                        print("⚠️  Данные для обучения не найдены в репозитории")
                        print(f"   Проверьте структуру репозитория.")
                        print(f"   Ожидается: workspace/prepared/features/*.csv или workspace/prepared/scalers/")
                        return False
                else:
                    print("⚠️  Структура workspace не найдена в репозитории")
                    print(f"   Проверьте структуру репозитория. Ожидается: workspace/")
                    return False
            finally:
                # Удаляем временную директорию
                if temp_dir.exists():
                    shutil.rmtree(temp_dir, ignore_errors=True)
            
            print(f"\n✓ Данные для обучения успешно скачаны!")
            print(f"  Локальная директория: {local_dir}")
            return True
            
        except Exception as e:
            print(f"❌ Ошибка при скачивании: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def download_hf_feature_analysis(self, local_dir: str = 'workspace/analysis-of-features') -> bool:
        """
        Скачивает результаты анализа фичей с Hugging Face
        
        Args:
            local_dir: Локальная директория для сохранения (по умолчанию: workspace/analysis-of-features)
        """
        print("=" * 60)
        print("Скачивание результатов анализа фичей с Hugging Face")
        print("=" * 60)
        
        print(f"📁 Репозиторий: {self.repo_id}")
        print(f"📂 Локальная директория: {local_dir}")
        
        try:
            local_path = Path(local_dir)
            local_path.mkdir(parents=True, exist_ok=True)
            
            # Скачиваем папку analysis-of-features и excluded_features.txt
            print(f"\nСкачивание данных (папка analysis-of-features/ и excluded_features.txt)...")
            temp_dir = Path('temp_hf_download')
            temp_dir.mkdir(exist_ok=True)
            
            try:
                downloaded_path = snapshot_download(
                    repo_id=self.repo_id,
                    repo_type="dataset",
                    local_dir=str(temp_dir),
                    token=self.token,
                    allow_patterns=[
                        "analysis-of-features/**",  # Скачиваем папку analysis-of-features
                        "workspace/excluded_features.txt"  # Скачиваем excluded_features.txt
                    ]
                )
                
                # Перемещаем данные из временной директории
                downloaded_path = Path(downloaded_path)
                analysis_source = downloaded_path / 'analysis-of-features'
                
                found_analysis = False
                if analysis_source.exists() and analysis_source.is_dir():
                    found_analysis = True
                    print(f"  Перемещение данных из временной директории...")
                    for item in analysis_source.iterdir():
                        dest = local_path / item.name
                        if item.is_file():
                            if dest.exists():
                                dest.unlink()
                            shutil.move(str(item), str(dest))
                        else:
                            if dest.exists():
                                shutil.rmtree(dest)
                            shutil.move(str(item), str(dest))
                
                # Перемещаем excluded_features.txt в workspace, если он найден
                excluded_source = downloaded_path / 'workspace' / 'excluded_features.txt'
                if excluded_source.exists():
                    workspace_path = Path('workspace')
                    workspace_path.mkdir(parents=True, exist_ok=True)
                    excluded_dest = workspace_path / 'excluded_features.txt'
                    if excluded_dest.exists():
                        excluded_dest.unlink()
                    shutil.move(str(excluded_source), str(excluded_dest))
                    print(f"  ✓ Перемещен excluded_features.txt в workspace/")
                
                if not found_analysis:
                    # Пробуем старый путь для обратной совместимости
                    old_analysis_source = downloaded_path / 'features-analysis'
                    if old_analysis_source.exists() and old_analysis_source.is_dir():
                        found_analysis = True
                        print(f"  Найдена папка features-analysis (старое название), перемещение...")
                        for item in old_analysis_source.iterdir():
                            dest = local_path / item.name
                            if item.is_file():
                                if dest.exists():
                                    dest.unlink()
                                shutil.move(str(item), str(dest))
                            else:
                                if dest.exists():
                                    shutil.rmtree(dest)
                                shutil.move(str(item), str(dest))
                
                if not found_analysis:
                    print("⚠️  Папка analysis-of-features не найдена в репозитории")
                    print(f"   Проверьте структуру репозитория. Ожидается: analysis-of-features/")
                    return False
            finally:
                # Удаляем временную директорию
                if temp_dir.exists():
                    shutil.rmtree(temp_dir, ignore_errors=True)
            
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
        description='Утилиты для работы с облачными сервисами (Hugging Face)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:

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
    hf_upload_features_parser.add_argument('--analysis-dir', type=str, default='workspace/analysis-of-features', help='Директория с результатами анализа')
    hf_upload_features_parser.add_argument('--commit-message', type=str, default='Upload feature analysis results', help='Сообщение коммита')
    
    # Hugging Face: Download feature analysis
    hf_download_features_parser = subparsers.add_parser('hf-download-features', help='Скачать результаты анализа фичей с Hugging Face')
    hf_download_features_parser.add_argument('--repo-id', type=str, required=True, help='ID репозитория (username/dataset-name)')
    hf_download_features_parser.add_argument('--token', type=str, default=None, help='Hugging Face токен (или HF_TOKEN env var)')
    hf_download_features_parser.add_argument('--local-dir', type=str, default='workspace/analysis-of-features', help='Локальная директория')
    
    # Hugging Face: Delete ticks
    hf_delete_ticks_parser = subparsers.add_parser('hf-delete-ticks', help='Удалить тики из Hugging Face датасета')
    hf_delete_ticks_parser.add_argument('--repo-id', type=str, required=True, help='ID репозитория (username/dataset-name)')
    hf_delete_ticks_parser.add_argument('--token', type=str, default=None, help='Hugging Face токен (или HF_TOKEN env var)')
    hf_delete_ticks_parser.add_argument('--commit-message', type=str, default='Delete tick data', help='Сообщение коммита')
    
    # Hugging Face: Delete training data
    hf_delete_training_parser = subparsers.add_parser('hf-delete-training', help='Удалить данные для обучения из Hugging Face датасета')
    hf_delete_training_parser.add_argument('--repo-id', type=str, required=True, help='ID репозитория (username/dataset-name)')
    hf_delete_training_parser.add_argument('--token', type=str, default=None, help='Hugging Face токен (или HF_TOKEN env var)')
    hf_delete_training_parser.add_argument('--include-scalers', action='store_true', default=True, help='Удалять scalers (по умолчанию: да)')
    hf_delete_training_parser.add_argument('--no-scalers', action='store_false', dest='include_scalers', help='Не удалять scalers')
    hf_delete_training_parser.add_argument('--include-cache', action='store_true', default=True, help='Удалять кэши (по умолчанию: да)')
    hf_delete_training_parser.add_argument('--no-cache', action='store_false', dest='include_cache', help='Не удалять кэши')
    hf_delete_training_parser.add_argument('--commit-message', type=str, default='Delete training data', help='Сообщение коммита')
    
    # Hugging Face: Delete feature analysis
    hf_delete_features_parser = subparsers.add_parser('hf-delete-features', help='Удалить результаты анализа фичей из Hugging Face датасета')
    hf_delete_features_parser.add_argument('--repo-id', type=str, required=True, help='ID репозитория (username/dataset-name)')
    hf_delete_features_parser.add_argument('--token', type=str, default=None, help='Hugging Face токен (или HF_TOKEN env var)')
    hf_delete_features_parser.add_argument('--commit-message', type=str, default='Delete feature analysis results', help='Сообщение коммита')
    
    # Hugging Face: Delete all data
    hf_delete_all_parser = subparsers.add_parser('hf-delete-all', help='Удалить все данные из Hugging Face датасета (очистить для новых загрузок)')
    hf_delete_all_parser.add_argument('--repo-id', type=str, required=True, help='ID репозитория (username/dataset-name)')
    hf_delete_all_parser.add_argument('--token', type=str, default=None, help='Hugging Face токен (или HF_TOKEN env var)')
    hf_delete_all_parser.add_argument('--commit-message', type=str, default='Delete all dataset data', help='Сообщение коммита')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    if args.command == 'hf-upload-ticks':
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
    
    elif args.command == 'hf-delete-ticks':
        if not HF_AVAILABLE:
            print("❌ huggingface_hub не установлен. Установите: pip install huggingface_hub")
            return
        deleter = HuggingFaceDeleter(repo_id=args.repo_id, token=args.token)
        deleter.delete_hf_ticks(commit_message=args.commit_message)
    
    elif args.command == 'hf-delete-training':
        if not HF_AVAILABLE:
            print("❌ huggingface_hub не установлен. Установите: pip install huggingface_hub")
            return
        deleter = HuggingFaceDeleter(repo_id=args.repo_id, token=args.token)
        include_scalers = getattr(args, 'include_scalers', True)
        include_cache = getattr(args, 'include_cache', True)
        deleter.delete_hf_training_data(
            include_scalers=include_scalers,
            include_cache=include_cache,
            commit_message=args.commit_message
        )
    
    elif args.command == 'hf-delete-features':
        if not HF_AVAILABLE:
            print("❌ huggingface_hub не установлен. Установите: pip install huggingface_hub")
            return
        deleter = HuggingFaceDeleter(repo_id=args.repo_id, token=args.token)
        deleter.delete_hf_feature_analysis(commit_message=args.commit_message)
    
    elif args.command == 'hf-delete-all':
        if not HF_AVAILABLE:
            print("❌ huggingface_hub не установлен. Установите: pip install huggingface_hub")
            return
        deleter = HuggingFaceDeleter(repo_id=args.repo_id, token=args.token)
        deleter.delete_all_data(commit_message=args.commit_message)
    
    print("\n" + "=" * 60)
    print("Готово!")
    print("=" * 60)


if __name__ == '__main__':
    main()

