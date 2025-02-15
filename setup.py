from setuptools import setup, find_packages
from pathlib import Path

PATH_ROOT = Path(__file__).parent.resolve()

def load_requirements(path_dir=PATH_ROOT, file_name="requirements.txt", comment_char="#"):
    with open(path_dir / file_name, "r", encoding="utf-8", errors="ignore") as file:
        lines = [line.rstrip() for line in file.readlines() if not line.startswith("#")]
    reqs = []
    for line in lines:
        if comment_char in line:
            line = line[: line.index(comment_char)].strip()
        if line.startswith("http"):
            continue
        if line:
            reqs.append(line)
    return reqs

setup(
    name='ragu',
    version='0.0.1',
    author='Mikhail Komarov',
    author_email='asphodel.bog.rem@gmail.com',
    description='Краткое описание пакета',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/AsphodelRem/RAGU.git',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    install_requires=[
        load_requirements()
    ],
)
