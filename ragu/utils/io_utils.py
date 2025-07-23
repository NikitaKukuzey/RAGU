from pathlib import Path
from typing import List
import json


def read_text_from_files(directory: str, file_extensions=None) -> List[str]:
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

    return texts

def read_text_from_chegeka(file: str) -> List[str]:
    """
    Чтение текстов из json-файла для ЧГК
    """
    with open(file, "r") as j_text:
        str_json = j_text.read()

    dct = json.loads(str_json)
    texts = []
    for el in dct:
        texts.append(el["page_content"])
    return texts

def read_qa_from_chegeka(file: str):
    with open(file, "r") as j_text:
        str_json = j_text.read()

    dct = json.loads(str_json)
    qas = []
    for el in dct:
        txt = el["inputs"]["text"]
        topic = el["inputs"]["topic"]
        question = el["instruction"].format(text=txt, topic=topic)
        qas.append((question, el["outputs"]))
    return qas