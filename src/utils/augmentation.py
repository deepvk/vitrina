import random
import numpy as np

from abc import abstractmethod, ABC

EXPECTED_NUMBER_OF_WORDS = 2
EXPECTED_NUMBER_OF_CHARS = 3
MAX_COUNT_AUGM = 2


class AugmentationWord(ABC):
    @abstractmethod
    def __call__(self, word: str) -> str:
        """
        Generates a new noisy word
        """

        raise NotImplementedError


class SwapAugmentation(AugmentationWord):
    """Swap one (random) pair of adjacent characters in a given word"""

    def __init__(self):
        pass

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

    def __init__(self, letters: dict, expected_number_of_chars: float):
        self.letters = letters
        self.expected_number_of_chars = expected_number_of_chars

    def __call__(self, word: str) -> str:
        symbols = []
        proba_per_char = (
            self.expected_number_of_chars / len(word) if len(word) >= self.expected_number_of_chars else 0.5
        )

        for ch in word:
            replace = np.random.binomial(1, proba_per_char)
            if replace and ch in self.letters.keys() and len(self.letters[ch]) != 0:
                random_symb = random.choice(self.letters[ch])
                symbols.append(random_symb)
            else:
                symbols.append(ch)
        return "".join(symbols)


class DiacriticsAugmentation(AugmentationWord):
    """
    Adds diacritics to chars of the word with a given probability
    """

    def __init__(self, expected_number_of_chars: float, count_adds: int = 1):
        self.expected_number_of_chars = expected_number_of_chars
        self.count_adds = count_adds

    def __call__(self, word: str) -> str:
        symbols = []
        proba_per_char = (
            self.expected_number_of_chars / len(word) if len(word) >= self.expected_number_of_chars else 0.5
        )
        for ch in word:
            replace = np.random.binomial(1, proba_per_char)
            if replace:
                char_with_diac = ch
                for i in range(self.count_adds):
                    randBytes = random.randint(0x300, 0x36F).to_bytes(2, "big")
                    char_with_diac += randBytes.decode("utf-16be")
                symbols.append(char_with_diac)
            else:
                symbols.append(ch)
        return "".join(symbols)


class SpaceAugmentation(AugmentationWord):
    """
    Adds spaces to chars of the word with a given probability
    """

    def __init__(self, expected_number_of_chars: float):
        self.expected_number_of_chars = expected_number_of_chars

    def __call__(self, word: str) -> str:
        symbols = []
        proba_per_char = (
            self.expected_number_of_chars / len(word) if len(word) >= self.expected_number_of_chars else 0.5
        )
        for ch in word:
            replace = np.random.binomial(1, proba_per_char)
            if replace:
                symbols.append(" ")
            symbols.append(ch)
        return "".join(symbols)


class AugmentationText:
    """
    Generates a noisy text with given parameters:
    text - original text to which word augmentations are applied
    proba_per_text - probability of noise for a given text;
    EXPECTED_NUMBER_OF_WORDS - expected value (average) of words in every text that we want to make noisy
    EXPECTED_NUMBER_OF_CHARS - expected value of chars in a word that we want to make noisy
    MAX_COUNT_AUGM - maximum value of augmentations that can be applied to every word
    """

    def __init__(self, leet_symbols: dict, cluster_symbols: dict, proba_per_text: float):
        diacritics = DiacriticsAugmentation(EXPECTED_NUMBER_OF_CHARS)
        clusters = SimilarCharAugmentation(cluster_symbols, EXPECTED_NUMBER_OF_CHARS)
        leet = SimilarCharAugmentation(leet_symbols, EXPECTED_NUMBER_OF_CHARS)
        spaces = SpaceAugmentation(EXPECTED_NUMBER_OF_CHARS)
        swap = SwapAugmentation()
        self.augmentations = np.array([diacritics, clusters, leet, spaces, swap])
        self.augmentations_probas = [0.3, 0.3, 0.3, 0.05, 0.05]  # sum must be equal to 1
        self.proba_per_text = proba_per_text

    def __call__(self, text: str) -> str:
        need_to_replace = np.random.binomial(1, self.proba_per_text)
        if not need_to_replace:
            return text

        words = text.split()
        result = ""

        proba_per_word = EXPECTED_NUMBER_OF_WORDS / len(words) if len(words) >= EXPECTED_NUMBER_OF_WORDS else 0.5
        for word in words:
            if len(word) == 0:
                continue

            replace = np.random.binomial(1, proba_per_word)
            if replace:
                number_augmentations = np.random.choice(range(1, MAX_COUNT_AUGM + 1))
                random_augmentations = np.random.choice(
                    self.augmentations, number_augmentations, p=self.augmentations_probas, replace=False
                )
                for augmentation in random_augmentations:
                    word = augmentation(word)
            result += word + " "

        return result.strip()
