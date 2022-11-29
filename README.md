# 👀 VITRina: VIsual Token Representations

[![Main](https://github.com/deepvk/vitrina/actions/workflows/main.yaml/badge.svg)](https://github.com/vitrina/actions/workflows/main.yaml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Checked with mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)

![](resources/images/vtr_architecture.jpg)


## Structure

- [`src`](./src) ‒ main source code with model and dataset implementations and code to train, test or infer model.
- [`notebooks`](./notebooks) ‒ notebooks with experiments and visualizations.
- [`scripts`](./scripts) ‒ different useful scripts, e.g. print dataset examples or evaluate existing models.
- [`tests`](./tests) ‒ unit tests.

## Requirements

Create virtual environment with `venv` or `conda` and install requirements:
```bash
pip install -r requirements.txt
```

For proper contributions, also use dev requirements:
```bash
pip install -r requirements-dev.txt
```

## Данные
[Toxic Russian Comments](https://www.kaggle.com/datasets/alexandersemiletov/toxic-russian-comments) ‒ датасет одноклассников для задачи multi-label классификации токсичных комментариев. 
Скачайте этот датасет и поместите в [`resources/data`](./resources/data).

Скрипт [`scripts/prepare_ok_dataset.py`](./scripts/prepare_ok_dataset.py) преобразует датасет в формат jsonl и заменяет метки класса на бинарное значение: 1 -- токсичный комментарий,
0 -- нейтральный комментарий.
```shell
python scripts/prepare_ok_dataset.py --data=resources/data/dataset.txt --save-to=resources/data/dataset.jsonl
```

## Скрипты

Все вспомогательные скрипты для подготовки необходимых для обучения инструментов расположены в [`scripts`](./scripts)

* [__scripts/clusterization.py__](./scripts/clusterization.py) ‒ скрипт для кластеризации изображений символов, поддерживаемых указанным шрифтом.
    ```shell
    python scripts/clusterization.py --font-path=resources/fonts/NotoSans.ttf --font-size=13 --clusters=500
    ```

* [__scripts/generate_noisy_dataset.py__](./scripts/generate_noisy_dataset.py) ‒ скрипт для зашумления датасета.
Уровень зашумления можно регулировать параметрами. `level-toxic` и `level-non-toxic` определяют множества, из которых берётся замена для символов с определённой вероятностью.
Значения этих параметров могут быть {1, 2, 3, 4}. Каждый следующий уровень включает в себя замены из предыдущего.
  * 1 ‒ замена на визуально похожие цифры ([resources/letter_replacement/letters1.json](resources/letter_replacement/letters1.json))
  * 2 ‒ замена на визуально похожие символы из стандартной раскладки клавиатуры или из латиницы ([resources/letter_replacement/letters2.json](resources/letter_replacement/letters2.json))
  * 3 ‒ замена на визуально похожие последовательности символов ([resources/letter_replacement/letters3.json](resources/letter_replacement/letters3.json))
  * 4 ‒ замена на символы из того же кластера (см. scripts/clusterization.py) ([resources/letter_replacement/clusterization.json](resources/letter_replacement/clusterization.json))
    
  ```shell
  python scripts/generate_noisy_dataset.py --data=resources/data/dataset.jsonl --level-toxic=4 --level-non-toxic=2 --p-toxic-word=0.5 --p-toxic-symbol=0.5 --p-space=0.01 --p-non-toxic-symbol=0.1 --save-to=resources/data/noisy_dataset.jsonl
   ```

* [__scripts/prepare_ok_dataset.py__](./scripts/prepare_ok_dataset.py) -- скрипт для преобразования датасета одноклассников в формат, с которым работает модель.

    `__label__INSULT скотина! что сказать` -> `{"text": "скотина! что сказать", "toxic": 1}`

    Выходной файл имеет разрешение jsonl и все метки в результате имеют бинарное значение.

    ```shell
    python scripts/prepare_ok_dataset.py --data=resources/data/dataset.txt --save-to=resources/data/dataset.jsonl
    ```

* [__scripts/train_tokenizer.py__](./scripts/train_tokenizer.py) ‒ скрипт для обучения WordPiece токенизатора. Используется для работы классического энкодера трансформера.
  ```shell
  python scripts/train_tokenizer.py --data=resources/data/noisy_dataset.jsonl --save-to=tokenizer
  ```
## Запуск обучения

Обучение запускается вызовом
```shell
python src/train.py
```

Изучите параметры запуска в файле скрипта [src/train.py](./src/train.py).

Рекомендуется запускать на gpu.

Для обучения необходимо указать аккаунт wandb. Используйте `wandb login`, чтобы войти в ваш аккаунт wandb.

Ознакомится с устройством свёрточной сети для получения эмбеддингов из визуального представления можно по изображению или коду [models/vtr/embedder.py](./models/vtr/embedder.py)
![](resources/images/conv_architecture.jpg)
