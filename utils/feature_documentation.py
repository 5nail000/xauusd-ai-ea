"""
Утилита для экспорта документации по фичам
Анализирует код генерации фичей и создает документацию
"""
import ast
import inspect
import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

def extract_feature_info_from_code() -> Dict[str, Dict]:
    """
    Извлекает информацию о фичах из исходного кода
    
    Returns:
        Словарь {feature_name: {description, formula, source_file, source_function}}
    """
    features_info = {}
    
    # Маппинг функций к файлам
    feature_modules = {
        'price_features': 'features/price_features.py',
        'technical_indicators': 'features/technical_indicators.py',
        'volatility_features': 'features/volatility_features.py',
        'time_features': 'features/time_features.py',
        'pattern_features': 'features/pattern_features.py',
        'multitimeframe_features': 'features/multitimeframe_features.py',
        'statistical_features': 'features/statistical_features.py',
        'volume_features': 'features/volume_features.py',
        'tick_features': 'features/tick_features.py'
    }
    
    # Парсим каждый модуль
    for module_name, filepath in feature_modules.items():
        if not Path(filepath).exists():
            continue
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                code = f.read()
            
            tree = ast.parse(code)
            
            # Находим все функции
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Извлекаем информацию о создаваемых фичах
                    func_features = _extract_features_from_function(node, code, module_name)
                    features_info.update(func_features)
        
        except Exception as e:
            print(f"⚠️  Ошибка при парсинге {filepath}: {e}")
            continue
    
    return features_info

def _extract_features_from_function(func_node: ast.FunctionDef, 
                                    source_code: str,
                                    module_name: str) -> Dict[str, Dict]:
    """
    Извлекает информацию о фичах из функции
    
    Args:
        func_node: AST узел функции
        source_code: Исходный код файла
        module_name: Название модуля
    
    Returns:
        Словарь с информацией о фичах
    """
    features = {}
    
    # Получаем docstring
    docstring = ast.get_docstring(func_node) or ""
    
    # Находим присваивания в DataFrame
    for node in ast.walk(func_node):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Subscript):
                    # df['feature_name'] = ...
                    if isinstance(target.value, ast.Name) and target.value.id in ['df', 'candles_df']:
                        if isinstance(target.slice, ast.Constant):
                            feature_name = target.slice.value
                        elif isinstance(target.slice, ast.Str):  # Python < 3.8
                            feature_name = target.slice.s
                        else:
                            continue
                        
                        # Пытаемся извлечь формулу
                        formula = _extract_formula(node.value, source_code)
                        
                        features[feature_name] = {
                            'description': _get_feature_description(feature_name, docstring),
                            'formula': formula,
                            'source_file': f'features/{module_name}.py',
                            'source_function': func_node.name,
                            'category': _get_feature_category(feature_name, module_name)
                        }
    
    return features

def _extract_formula(value_node: ast.AST, source_code: str) -> Optional[str]:
    """
    Извлекает формулу из AST узла
    
    Args:
        value_node: AST узел значения
        source_code: Исходный код
    
    Returns:
        Строка с формулой или None
    """
    try:
        # Пытаемся получить строковое представление
        if isinstance(value_node, ast.BinOp):
            # Бинарная операция
            left = _ast_to_string(value_node.left, source_code)
            right = _ast_to_string(value_node.right, source_code)
            op = _get_operator(value_node.op)
            return f"{left} {op} {right}"
        elif isinstance(value_node, ast.Call):
            # Вызов функции
            func_name = _ast_to_string(value_node.func, source_code)
            args = [_ast_to_string(arg, source_code) for arg in value_node.args]
            return f"{func_name}({', '.join(args)})"
        else:
            return _ast_to_string(value_node, source_code)
    except:
        return None

def _ast_to_string(node: ast.AST, source_code: str) -> str:
    """Преобразует AST узел в строку"""
    try:
        if isinstance(node, ast.Constant):
            return str(node.value)
        elif isinstance(node, ast.Str):  # Python < 3.8
            return node.s
        elif isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{_ast_to_string(node.value, source_code)}.{node.attr}"
        elif isinstance(node, ast.Subscript):
            value = _ast_to_string(node.value, source_code)
            if isinstance(node.slice, ast.Constant):
                slice_val = node.slice.value
            elif isinstance(node.slice, ast.Str):
                slice_val = node.slice.s
            else:
                slice_val = "..."
            return f"{value}['{slice_val}']"
        else:
            # Для сложных выражений возвращаем "..."
            return "..."
    except:
        return "..."

def _get_operator(op_node: ast.operator) -> str:
    """Преобразует оператор в строку"""
    op_map = {
        ast.Add: '+',
        ast.Sub: '-',
        ast.Mult: '*',
        ast.Div: '/',
        ast.Mod: '%',
        ast.Pow: '**'
    }
    return op_map.get(type(op_node), '?')

def _get_feature_description(feature_name: str, docstring: str) -> str:
    """
    Извлекает описание фичи из docstring или генерирует по названию
    
    Args:
        feature_name: Название фичи
        docstring: Docstring функции
    
    Returns:
        Описание фичи
    """
    # Пытаемся найти описание в docstring
    if docstring:
        lines = docstring.split('\n')
        for line in lines:
            if feature_name.lower() in line.lower():
                return line.strip()
    
    # Генерируем описание по названию
    return _generate_description_from_name(feature_name)

def _generate_description_from_name(feature_name: str) -> str:
    """Генерирует описание фичи на основе её названия"""
    descriptions = {
        'returns': 'Процентное изменение цены закрытия',
        'log_returns': 'Логарифмическая доходность',
        'rsi': 'Индекс относительной силы (RSI)',
        'sma': 'Простое скользящее среднее (SMA)',
        'ema': 'Экспоненциальное скользящее среднее (EMA)',
        'macd': 'Схождение-расхождение скользящих средних (MACD)',
        'atr': 'Средний истинный диапазон (ATR)',
        'bb': 'Полосы Боллинджера',
        'stoch': 'Стохастический осциллятор',
        'cci': 'Индекс товарного канала (CCI)',
        'williams': 'Индекс Уильямса %R',
        'adx': 'Индекс среднего направленного движения (ADX)',
        'momentum': 'Моментум',
        'roc': 'Скорость изменения (ROC)',
        'volume': 'Объем',
        'obv': 'Балансовый объем (OBV)',
        'vwap': 'Средневзвешенная цена по объему (VWAP)',
        'hour': 'Час дня',
        'day_of_week': 'День недели',
        'month': 'Месяц',
        'lag': 'Лаг (задержка)',
        'rolling_mean': 'Скользящее среднее',
        'rolling_std': 'Скользящее стандартное отклонение',
        'zscore': 'Z-score нормализация',
        'position': 'Позиция в диапазоне',
        'distance': 'Расстояние до',
        'spread': 'Спред',
        'shadow': 'Тень свечи',
        'body': 'Тело свечи',
        'doji': 'Паттерн Doji',
        'hammer': 'Паттерн Hammer',
        'engulfing': 'Паттерн Engulfing',
        'tick': 'Тиковые данные',
        'second': 'Секундные свечи'
    }
    
    feature_lower = feature_name.lower()
    for key, desc in descriptions.items():
        if key in feature_lower:
            # Извлекаем параметры из названия
            parts = feature_name.split('_')
            params = [p for p in parts if p.isdigit()]
            if params:
                return f"{desc} (период: {', '.join(params)})"
            return desc
    
    return f"Фича: {feature_name}"

def _get_feature_category(feature_name: str, module_name: str) -> str:
    """Определяет категорию фичи"""
    category_map = {
        'price_features': 'Ценовые фичи',
        'technical_indicators': 'Технические индикаторы',
        'volatility_features': 'Волатильность',
        'time_features': 'Временные фичи',
        'pattern_features': 'Свечные паттерны',
        'multitimeframe_features': 'Мультитаймфреймовые фичи',
        'statistical_features': 'Статистические фичи',
        'volume_features': 'Объемные фичи',
        'tick_features': 'Тиковые фичи'
    }
    
    return category_map.get(module_name, 'Прочие')

def create_feature_documentation(feature_columns: List[str],
                                scaler_path: Optional[str] = None,
                                output_path: Optional[str] = None) -> Dict:
    """
    Создает документацию по фичам
    
    Args:
        feature_columns: Список названий фичей
        scaler_path: Путь к scaler файлу (для получения статистики)
        output_path: Путь для сохранения документации
    
    Returns:
        Словарь с документацией
    """
    print("Создание документации по фичам...")
    
    # Извлекаем информацию из кода
    features_info = extract_feature_info_from_code()
    
    # Загружаем статистику из scaler, если доступна
    feature_stats = {}
    if scaler_path and Path(scaler_path).exists():
        try:
            import pickle
            with open(scaler_path, 'rb') as f:
                scaler_data = pickle.load(f)
                feature_stats = scaler_data.get('feature_stats', {})
        except Exception as e:
            print(f"⚠️  Не удалось загрузить статистику из scaler: {e}")
    
    # Создаем документацию для каждого фича
    documentation = {
        'metadata': {
            'created_at': datetime.now().isoformat(),
            'total_features': len(feature_columns),
            'scaler_path': scaler_path
        },
        'features': {}
    }
    
    for feature_name in feature_columns:
        feature_info = features_info.get(feature_name, {})
        stats = feature_stats.get(feature_name, {})
        
        documentation['features'][feature_name] = {
            'name': feature_name,
            'description': feature_info.get('description', _generate_description_from_name(feature_name)),
            'formula': feature_info.get('formula'),
            'category': feature_info.get('category', 'Неизвестно'),
            'source_file': feature_info.get('source_file'),
            'source_function': feature_info.get('source_function'),
            'statistics': {
                'mean': stats.get('mean'),
                'std': stats.get('std')
            } if stats else None
        }
    
    # Сохраняем документацию
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Сохраняем в JSON
        json_path = output_path.with_suffix('.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(documentation, f, indent=2, ensure_ascii=False)
        
        # Сохраняем в Markdown
        md_path = output_path.with_suffix('.md')
        _save_markdown_documentation(documentation, md_path)
        
        print(f"✓ Документация сохранена:")
        print(f"  - {json_path}")
        print(f"  - {md_path}")
    
    return documentation

def _save_markdown_documentation(doc: Dict, output_path: Path):
    """Сохраняет документацию в формате Markdown"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Документация по фичам\n\n")
        f.write(f"**Дата создания:** {doc['metadata']['created_at']}\n\n")
        f.write(f"**Всего фичей:** {doc['metadata']['total_features']}\n\n")
        
        if doc['metadata'].get('scaler_path'):
            f.write(f"**Scaler:** {doc['metadata']['scaler_path']}\n\n")
        
        f.write("---\n\n")
        
        # Группируем по категориям
        categories = {}
        for feature_name, feature_info in doc['features'].items():
            category = feature_info.get('category', 'Прочие')
            if category not in categories:
                categories[category] = []
            categories[category].append((feature_name, feature_info))
        
        # Выводим по категориям
        for category in sorted(categories.keys()):
            f.write(f"## {category}\n\n")
            
            for feature_name, feature_info in sorted(categories[category]):
                f.write(f"### {feature_name}\n\n")
                
                if feature_info.get('description'):
                    f.write(f"**Описание:** {feature_info['description']}\n\n")
                
                if feature_info.get('formula'):
                    f.write(f"**Формула:** `{feature_info['formula']}`\n\n")
                
                if feature_info.get('source_file'):
                    f.write(f"**Источник:** `{feature_info['source_file']}` → `{feature_info.get('source_function', 'N/A')}`\n\n")
                
                if feature_info.get('statistics'):
                    stats = feature_info['statistics']
                    if stats.get('mean') is not None:
                        f.write(f"**Статистика (из обучающей выборки):**\n")
                        f.write(f"- Среднее: {stats['mean']:.6f}\n")
                        if stats.get('std'):
                            f.write(f"- Стандартное отклонение: {stats['std']:.6f}\n")
                        f.write("\n")
                
                f.write("---\n\n")
        
        # Список всех фичей
        f.write("## Полный список фичей\n\n")
        for feature_name in sorted(doc['features'].keys()):
            f.write(f"- `{feature_name}`\n")

def export_feature_documentation_for_model(model_type: str,
                                          feature_columns: List[str],
                                          scaler_path: Optional[str] = None) -> str:
    """
    Экспортирует документацию по фичам для модели
    
    Args:
        model_type: Тип модели ('encoder' или 'timeseries')
        feature_columns: Список фичей
        scaler_path: Путь к scaler файлу
    
    Returns:
        Путь к сохраненной документации
    """
    output_path = f'workspace/models/checkpoints/{model_type}_model_features_documentation'
    
    documentation = create_feature_documentation(
        feature_columns=feature_columns,
        scaler_path=scaler_path,
        output_path=output_path
    )
    
    return output_path

