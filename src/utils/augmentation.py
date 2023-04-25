import random
import numpy as np

from abc import abstractmethod, ABC


class AugmentationWord(ABC):
    @abstractmethod
    def __call__(self, *args, **kwargs):
        """
        Generates a new noisy word
        """

        raise NotImplementedError


class SwapAugmentation(AugmentationWord):
    """Swap one (random) pair of adjacent characters in a given word"""

    def __init__(self):
        pass

    def __call__(self, word: str, proba_per_char: float) -> str:
        if len(word) < 2:
            return word

        ind = np.random.choice(len(word) - 1)
        return word[:ind] + word[ind + 1] + word[ind] + word[ind + 2 :]


class LetterAugmentation(AugmentationWord):
    """
    Changes chars in the word by some visually similar characters (l33tspeak or clusters)
    with a given probability
    """

    def __init__(self, letters: dict):
        self.letters = letters

    def __call__(self, word: str, proba_per_char: float) -> str:
        symbols = []
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

    def __init__(self):
        pass

    def __call__(self, word: str, proba_per_char: float, count_adds: int = 1) -> str:
        symbols = []
        for ch in word:
            replace = np.random.binomial(1, proba_per_char)
            if replace:
                char_with_diac = ch
                for i in range(count_adds):
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

    def __init__(self):
        pass

    def __call__(self, word: str, proba_per_char: float) -> str:
        symbols = []
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
    proba_per_text - probability of noise for a given text
    expected_words - expected value (average) of words in every text that we want to make noisy
    expected_chars - expected value of chars in a word that we want to make noisy
    max_count_augm - maximum value of augmentations that can be applied to every word
    """

    def __init__(self, leet_symbols: dict, cluster_symbols: dict):
        diacritics = DiacriticsAugmentation()
        spaces = SpaceAugmentation()
        clusters = LetterAugmentation(cluster_symbols)
        leet = LetterAugmentation(leet_symbols)
        swap = SwapAugmentation()
        self.augmentations = np.array([diacritics, clusters, leet, spaces, swap])
        self.augmentations_probas = [0.3, 0.3, 0.3, 0.05, 0.05]  # sum must be equal to 1

    def __call__(self, text: str, proba_per_text=0.8, expected_words=3, expected_chars=2, max_count_augm=2) -> str:
        need_to_replace = np.random.binomial(1, proba_per_text)
        if not need_to_replace:
            return text

        words = text.split()
        result = ""
        proba_per_word = expected_words / len(words)
        for word in words:
            replace = np.random.binomial(1, proba_per_word)
            if replace:
                number_augmentations = np.random.choice(range(1, max_count_augm + 1))
                random_augmentations = np.random.choice(
                    self.augmentations, number_augmentations, p=self.augmentations_probas, replace=False
                )
                proba_per_char = expected_chars / len(word)
                for augmentation in random_augmentations:
                    word = augmentation(word, proba_per_char)
            result += word + " "

        return result.strip()
