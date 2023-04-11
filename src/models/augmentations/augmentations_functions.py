import random
import numpy as np


class SwapAugmentation:
    def __init__(self):
        pass

    def __call__(self, text, proba):

        if len(text) < 2:
            return text
        words = text.split()
        symbols = []
        for word in words:

            for i in range(0, len(word) - 1, 2):
                replace = np.random.binomial(1, proba)
                if replace:
                    symbols += [word[i + 1], word[i]]
                else:
                    symbols += [word[i], word[i + 1]]

            if len(word) % 2 == 1:
                symbols.append(word[-1])

            symbols.append(" ")

        return "".join(symbols).strip()


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


class LetterAugmentation:  # Leet or Clusters
    def __init__(self, letters):
        self.letters = letters

    def __call__(self, text, proba):
        symbols = []
        for ch in text:
            replace = np.random.binomial(1, proba)
            if replace and ch in self.letters.keys() and len(self.letters[ch]) != 0:
                random_symb = random.choice(self.letters[ch])
                symbols.append(random_symb)
            else:
                symbols.append(ch)
        return "".join(symbols)


class ProbelsAugmentation:
    def __init__(self):
        pass

    def __call__(self, text, proba):
        symbols = []
        for ch in text:
            replace = np.random.binomial(1, proba)
            if replace and ch != " ":
                symbols.append(" ")
            symbols.append(ch)
        return "".join(symbols)
