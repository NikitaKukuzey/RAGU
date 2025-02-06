from pathlib import Path


def read_text_from_files(directory: str, file_extensions=None):
    """
    Чтение текста из всех файлов в директории и её поддиректориях.

    :param directory: Путь к директории для обхода (объект Path).
    :param file_extensions: Список расширений файлов для фильтрации (например, ['.txt', '.md']).
    :return: Список строк, каждая из которых — содержимое файла.
    """
    texts = []
    directory = Path(directory)
    for file_path in directory.rglob('*'):
        if file_path.is_file() and (file_extensions is None or file_path.suffix in file_extensions):
            try:
                with file_path.open('r', encoding='utf-8') as f:
                    texts.append(f.read())
            except (UnicodeDecodeError, PermissionError) as e:
                print(f"⚠️ Не удалось прочитать {file_path}: {e}")

    return '/n'.join(texts)