"""
Утилита для работы с исключаемыми фичами
Читает список фичей для исключения из файла excluded_features.txt
"""
from pathlib import Path
from typing import List, Optional

DEFAULT_EXCLUSIONS_FILE = Path('workspace/excluded_features.txt')

def load_excluded_features(exclusions_file: Optional[Path] = None) -> List[str]:
    """
    Загружает список фичей для исключения из файла
    
    Args:
        exclusions_file: Путь к файлу со списком исключений.
                        Если None, используется DEFAULT_EXCLUSIONS_FILE
    
    Returns:
        Список названий фичей для исключения (пустой список, если файл не найден или пуст)
    """
    if exclusions_file is None:
        exclusions_file = DEFAULT_EXCLUSIONS_FILE
    
    excluded_features = []
    
    # Проверяем существование файла
    if not exclusions_file.exists():
        return excluded_features
    
    # Читаем файл
    try:
        with open(exclusions_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # Пропускаем пустые строки и комментарии
                if line and not line.startswith('#'):
                    excluded_features.append(line)
    except Exception as e:
        print(f"⚠️  Предупреждение: Не удалось прочитать файл исключений {exclusions_file}: {e}")
        return []
    
    return excluded_features

def save_excluded_features(features: List[str], 
                          exclusions_file: Optional[Path] = None,
                          add_header: bool = True):
    """
    Сохраняет список фичей для исключения в файл
    
    Args:
        features: Список названий фичей для исключения
        exclusions_file: Путь к файлу для сохранения.
                        Если None, используется DEFAULT_EXCLUSIONS_FILE
        add_header: Добавлять ли заголовок с комментарием
    """
    if exclusions_file is None:
        exclusions_file = DEFAULT_EXCLUSIONS_FILE
    
    # Создаем директорию, если её нет
    exclusions_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Сохраняем файл
    try:
        with open(exclusions_file, 'w', encoding='utf-8') as f:
            if add_header:
                f.write("# Список фичей для исключения из обучения\n")
                f.write("# Одна фича на строку\n")
                f.write("# Строки, начинающиеся с #, игнорируются\n")
                f.write("# Пустые строки игнорируются\n\n")
            
            for feature in sorted(set(features)):  # Убираем дубликаты и сортируем
                f.write(f"{feature}\n")
    except Exception as e:
        print(f"❌ Ошибка при сохранении файла исключений {exclusions_file}: {e}")
        raise

