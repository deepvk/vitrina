import random
import numpy as np

from abc import abstractmethod, ABC

PROBA_FOR_SHORT_WORDS = 0.0
PROBA_FOR_SHORT_TEXTS = 0.0


class AugmentationWord(ABC):
    @abstractmethod
    def __call__(self, word: str) -> str:
        """
        Generates a new noisy word
        """

        raise NotImplementedError


class SwapAugmentation(AugmentationWord):
    """Swap one (random) pair of adjacent characters in a given word"""

    def __call__(self, word: str) -> str:
        if len(word) < 2:
            return word

        ind = np.random.choice(len(word) - 1)
        return word[:ind] + word[ind + 1 : ind - 1 : -1] + word[ind + 2 :]


class SimilarCharAugmentation(AugmentationWord):
    """
    Changes chars in the word by some visually similar characters (l33tspeak or clusters)
    with a given probability
    """

    def __init__(self, letters: dict, expected_changes_per_word: float):
        self.letters = letters
        self.expected_changes_per_word = expected_changes_per_word

    def __call__(self, word: str) -> str:
        symbols = list(word)
        proba_per_char = (
            self.expected_changes_per_word / len(word)
            if len(word) >= self.expected_changes_per_word
            else PROBA_FOR_SHORT_WORDS
        )

        for i, ch in enumerate(symbols):
            replace = np.random.binomial(1, proba_per_char)
            if replace and ch in self.letters.keys() and len(self.letters[ch]) != 0:
                random_symbol = random.choice(self.letters[ch])
                symbols[i] = random_symbol

        return "".join(symbols)


class DiacriticsAugmentation(AugmentationWord):
    """
    Adds diacritics to chars of the word with a given probability
    """

    def __init__(self, expected_changes_per_word: float, diacritics_per_char: int = 1):
        self.expected_changes_per_word = expected_changes_per_word
        self.diacritics_per_char = diacritics_per_char

    def __call__(self, word: str) -> str:
        symbols = list(word)

        proba_per_char = (
            self.expected_changes_per_word / len(word)
            if len(word) >= self.expected_changes_per_word
            else PROBA_FOR_SHORT_WORDS
        )
        for index, ch in enumerate(symbols):
            replace = np.random.binomial(1, proba_per_char)
            if not replace:
                continue
            for i in range(self.diacritics_per_char):
                randBytes = random.randint(0x300, 0x36F).to_bytes(2, "big")
                symbols[index] += randBytes.decode("utf-16be")

        return "".join(symbols)


class SpaceAugmentation(AugmentationWord):
    """
    Adds spaces to chars of the word with a given probability
    """

    def __init__(self, expected_changes_per_word: float):
        self.expected_changes_per_word = expected_changes_per_word

    def __call__(self, word: str) -> str:
        symbols = []
        proba_per_char = (
            self.expected_changes_per_word / len(word)
            if len(word) >= self.expected_changes_per_word
            else PROBA_FOR_SHORT_WORDS
        )
        for ch in word:
            replace = np.random.binomial(1, proba_per_char)
            if replace:
                symbols.append(" ")
            symbols.append(ch)
        return "".join(symbols)


class IdAugmentation(AugmentationWord):
    """
    Doesn't apply any transformation
    """

    def __call__(self, word: str) -> str:
        return word


def init_augmentations(expected_changes_per_word, cluster_symbols, leet_symbols):
    diacritics = DiacriticsAugmentation(expected_changes_per_word)
    clusters = SimilarCharAugmentation(cluster_symbols, expected_changes_per_word)
    leet = SimilarCharAugmentation(leet_symbols, expected_changes_per_word)
    spaces = SpaceAugmentation(expected_changes_per_word)
    swap = SwapAugmentation()
    augmentations = [(diacritics, 0.28), (clusters, 0.28), (leet, 0.28), (spaces, 0.05), (swap, 0.05)]
    return augmentations


class TextAugmentationWrapper:
    """
    Generates a noisy text with given parameters:

    :param text - original text to which word augmentations are applied
    :param proba_per_text - probability of noise for a given text;
    :param expected_changes_per_text - expected value (average) of words in every text that we want to make noisy
    :param augmentations - augmentations of words with probabilities
    :param max_augmentations - maximum value of augmentations that can be applied to every word
    """

    def __init__(
        self,
        augmentations: list[tuple[AugmentationWord, float]],
        proba_per_text: float = 1,
        expected_changes_per_text: int = 3,
        max_augmentations: int = 2,
    ):
        self.augmentations, self.probas = zip(*augmentations)
        self.augmentations += (IdAugmentation(),)
        self.probas += (1 - sum(self.probas),)
        self.proba_per_text = proba_per_text
        self.expected_changes_per_text = expected_changes_per_text
        self.max_augmentations = max_augmentations

    def __call__(self, text: str) -> str:
        need_to_replace = np.random.binomial(1, self.proba_per_text)
        if not need_to_replace:
            return text

        words = text.split()

        proba_per_word = (
            self.expected_changes_per_text / len(words)
            if len(words) >= self.expected_changes_per_text
            else PROBA_FOR_SHORT_TEXTS
        )
        for i, word in enumerate(words):
            if len(word) == 0:
                continue

            replace = np.random.binomial(1, proba_per_word)
            if not replace:
                continue

            random_augmentations = np.random.choice(
                self.augmentations, self.max_augmentations, p=self.probas, replace=False
            )
            for augmentation in random_augmentations:
                word = augmentation(word)

            words[i] = word

        return " ".join(words)
