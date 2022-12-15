from typing import TypedDict


class DatasetSample(TypedDict):
    """Type definition for a sample in a sequence classification dataset.
    Example:
        {"text": "скотина! что сказать", "label": 1}
    """

    text: str
    label: int


class SLDatasetSample(TypedDict):
    """Type definition for a sample in a sequence labeling dataset.
    Example:
        {"text": [("cкотина", 0), ("!", 0), ("что", 0), ("сказать", 0)], "label": 1}
    """

    text: list[tuple[str, int]]
    label: int
