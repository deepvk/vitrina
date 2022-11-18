# 👀 VITRina: VIsual Token Representations

[![Main](https://github.com/deepvk/vitrina/actions/workflows/main.yaml/badge.svg)](https://github.com/vitrina/actions/workflows/main.yaml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Checked with mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)

## Данные
[Toxic Russian Comments](https://www.kaggle.com/datasets/alexandersemiletov/toxic-russian-comments) -- датасет одноклассников для задачи multi-label классификации токсичных комментариев. 
Скачайте этот датасет и поместите в [`resources/data`](./resources/data).

Скрипт [`scripts/prepare_ok_dataset.py`](./scripts/prepare_ok_dataset.py) преобразует датасет в формат jsonl и заменяет метки класса на бинарное значение: 1 -- токсичный комментарий,
0 -- нейтральный комментарий.
```angular2html
python scripts/prepare_ok_dataset.py --data=resources/data/dataset.txt --save-to=resources/data/dataset.jsonl
```

## Скрипты

Все вспомогательные скрипты для подготовки необходимых для обучения инструментов расположены в [`scripts`](./scripts)

[__scripts/clusterization.py__](./scripts/clusterization.py) -- скрипт для кластеризации изображений символов, поддерживаемых указанным шрифтом.

_Параметры:_
* `--font-path` -- путь к шрифту, которым необходимо генерировать изображения (_default_: fonts/NotoSans.ttf)
* `--font-size` -- размер шрифта в пикселях (_default_: 13)
* `--clusters` -- количество кластеров, на которые разбивать множество изображений символов (_default_: 500)
```angular2html
python scripts/clusterization.py --font-path=resources/fonts/NotoSans.ttf --font-size=13 --clusters=500
```

[__scripts/generate_noisy_dataset.py__](./scripts/generate_noisy_dataset.py) -- скрипт для зашумления датасета.
Уровень зашумления можно регулировать параметрами. `level-toxic` и `level-non-toxic` определяют множества, из которых берётся замена для символов с определённой вероятностью.
Значения этих параметров могут быть {1, 2, 3, 4}. Каждый следующий уровень включает в себя замены из предыдущего.

1 -- замена на визуально похожие цифры ([resources/letter_replacement/letters1.json](resources/letter_replacement/letters1.json))

2 -- замена на визуально похожие символы из стандартной раскладки клавиатуры или из латиницы ([resources/letter_replacement/letters2.json](resources/letter_replacement/letters2.json))

3 -- замена на визуально похожие последовательности символов ([resources/letter_replacement/letters3.json](resources/letter_replacement/letters3.json))

4 -- замена на символы из того же кластера (см. scripts/clusterization.py) ([resources/letter_replacement/clusterization.json](resources/letter_replacement/clusterization.json))

_Параметры:_
* `--data` -- путь к исходному датасету
* `--level-toxic` -- уровень зашумления токсичных слов (_default_: 4)
* `--level-non-toxic` -- уровень зашумления нетоксичных слов (_default_: 2)
* `--p-toxic-word` -- вероятность внесения изменений в токсичное слово (_default_: 0.5)
* `--p-toxic-symbol` -- вероятность изменения символа (_default_: 0.5)
* `--p-non-toxic-symbol` -- вероятность изменения символа в нетоксичном слове (_default_: 0.1)
* `--p-space` -- вероятность добавления пробела после символа (_default_: 0.01)
* `--sl` -- если флаг выставлен, то будет сгенерирован датасет для задачи sequence labeling (разметки оскорбительных слов)
* `--save-to` -- путь к файлу, в котором будет сохранён зашумлённый датасет (_default_: resources/data/noisy_dataset.jsonl)
```angular2html
python scripts/generate_noisy_dataset.py --data=resources/data/dataset.jsonl --level-toxic=4 --level-non-toxic=2 --p-toxic-word=0.5 --p-toxic-symbol=0.5 --p-space=0.01 --p-non-toxic-symbol=0.1 --save-to=resources/data/noisy_dataset.jsonl
```

[__scripts/prepare_ok_dataset.py__](./scripts/prepare_ok_dataset.py) -- скрипт для преобразования датасета одноклассников в формат, с которым работает модель.

`__label__INSULT скотина! что сказать` -> `{"text": "скотина! что сказать", "toxic": 1}`

Выходной файл имеет разрешение jsonl и все метки в результате имеют бинарное значение.

_Параметры:_
* `--data` -- путь к исходному датасету (_default_: resources/data/dataset.txt)
* `--save-to` -- файл, в котором будет сохранён результат (_default_: resources/data/ok_toxic.jsonl)

```angular2html
python scripts/prepare_ok_dataset.py --data=resources/data/dataset.txt --save-to=resources/data/dataset.jsonl
```

[__scripts/train_tokenizer.py__](./scripts/train_tokenizer.py) -- скрипт для обучения WordPiece токенизатора. Используется для работы классического энкодера трансформера.

_Так как мы хотим обучить токенизатор только на тренировочных данных, при этом датасет не разбивается на отдельные файлы для обучения, тистирования и валидации, 
необходимо передать размер тренировочной и валидационной выборок, а так же random-state с которым происходит это разбиение в [`src/train.py`](./src/train.py).
На основании этих параметров скрипт разделит выборку и обучит токенизатр только на тренировочной части._
__TODO__: исправить (разбить датасет на отдельные файлы)

_Параметры:_
* `--data` -- данные, на которых обучается токенизатор (_default:_ resources/data/noisy_dataset.jsonl)
* `--test-size` -- размер тестовой выборки (_default:_ 0.1)
* `--val-size` -- размер валидационной выборки (_default:_ 0.1)
* `--random-state` (_default:_ 21)
```angular2html
python scripts/train_tokenizer.py --data=resources/data/noisy_dataset.jsonl --test-size=0.1 --val-size=0.1 --random-state=21 --save-to=tokenizer
```
## Запуск обучения

Обучение запускается вызовом
```angular2html
python src/train.py
```

_Параметры:_
* `--data` -- путь к данным для обучения (_default:_ resources/data/noisy_dataset.jsonl)
* `--max-seq-len` -- используется при создании датасетов. В случае vtr -- ограничение на количество слайсов, в случае классического трансформера -- на количество токенов. (_default_: 512)
* `--epochs` -- количество эпох обучения (_default:_ 10)
* `--test-size` -- размер тестовой выборки (_default:_ 0.1)
* `--val-size` -- размер валидационной выборки (_default:_ 0.1)
* `--random-state` (_default:_ 21)
* `--log-every` -- как часто пересчитывать значение метрик на валидационной выборки. Эти значения логируются в wandb (_default:_ 1000)
* `--num-workers` -- количество потоков в DataLoader (_default:_ 1)
* `--save-to` -- путь, по которому сохраняются веса модели после обучения, если значение указано, если нет, то веса не сохраняются.
* `--beta1`, `--beta2` -- параметры оптимизатора Adam (_default:_ 0.9, 0.999)
* `--sl` -- если выставлен, значит решается задача разметки последовательности
* `--test-data` -- если указан, то исходный датасет не разбивается на обучающую и тестовую выборки. Качество меряется на датасете, которые лежит по пути `test-data` (_default:_ None)
* `--val` -- если выставлен, то логируется качество на валидационной выборке, иначе на тестовой (TODO: переделать)
* `--dropout` -- вероятность dropout
* `--device` -- cuda или cpu
* `--warmup` -- количество шагов для разогрева Linear Scheduler (_default:_ 1000)
* `--batch-size` -- размер батча (_default:_ 32)
* `--lr` -- learning rate (_default:_ 5e-5)

_Параметры для обучения vtr:_
* `--max-slices-count-per-word` -- используется для ограничения на количество слайсов для одного слова. Подробнее изучить процесс обработки последовательности слов можно взглянув на изображение (_default:_ 9)
![](`r https://sun9-43.userapi.com/impg/frHiG2LjSyGa9b3hVmhrDx678d3yoE6gAahe-w/kO7oP431eEg.jpg?size=2298x1322&quality=95&sign=fa872fbd34c5598b417954219cbdc076&type=album`)
* `--font-size` -- размер шрифта, с которым генерируется изображение (_default:_ 15)
* `--window-size` -- размер окна (_default:_ 30)
* `--stride` -- размер шага окна
* `--emb-size` -- размер эмбеддинга, получаемый визуальными представлениями (_default:_ 768)
* `--font` -- шрифт, которым генерируются изображения символов
* `--out-channels` -- число выходных каналов в последнем свёрточном слое (_default:_ 256)
* `--nhead` -- количество голов слоя трансформера (_default:_ 12)
* `--kernel-size` -- размер ядра свёртки (_default:_ 3)
* `--num-layers` -- количество слоёв энкодера трансформера в модели визуальных представлений (_default:_ 1)

_Параметры для обучения одного слоя трансформера:_
* `--tokenizer` -- путь к токенизатору, созданному скриптом [`scripts/train_tokenizer.py`](./scripts/train_tokenizer.py). (_default:_ )


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

