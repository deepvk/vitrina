import random
import numpy as np


class LetterAugmentation:  # Leet or Clusters
    def __init__(self, letters):
        self.letters = letters

    def __call__(self, char):
        symbols = []
        if char in self.letters.keys() and len(self.letters[char]) != 0:
            random_symb = random.choice(self.letters[char])
            return random_symb
        return char


class DiacriticsAugmentation:
    def __init__(self):
        pass

    def __call__(self, char, count_adds=1):
        char_with_diac = char
        for i in range(count_adds):
            randBytes = random.randint(0x300, 0x36F).to_bytes(2, "big")
            char_with_diac += randBytes.decode("utf-16be")
        return char_with_diac


class ProbelAugmentation:
    def __init__(self):
        pass

    def __call__(self, char):
        return char + " "


class CamAugmentation:
    def __init__(self):
        pass

    def __call__(self, text):
        if len(text) < 4:
            return text

        words = text.split()
        symbols = []
        for word in words:
            random_permutation = np.random.permutation(list(word[1:-1])).tolist()
            symbols += [word[0]] + random_permutation + [word[-1], " "]
        return "".join(symbols).strip()


class Augmentation:
    def __init__(self, leet, clusters):
        diacritics = DiacriticsAugmentation()
        probels = ProbelAugmentation()
        clusters = LetterAugmentation(clusters)
        leet = LetterAugmentation(leet)
        self.augmentations = [diacritics, probels, clusters, leet]

    def __call__(self, text, proba_per_text, proba_per_char):
        need_to_replace = np.random.binomial(1, proba_per_text)
        if not need_to_replace:
            return text

        symbols = []
        for ch in text:
            replace = np.random.binomial(1, proba_per_char)
            if replace and ch != " ":
                random_augm = random.choice(self.augmentations)
                symbols.append(random_augm(ch))
            else:
                symbols.append(ch)
        return "".join(symbols)
