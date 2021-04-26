import numpy as np
from nltk.tokenize import word_tokenize


class Preprocessing():
    def __init__(self, X, cols, stop_words):
        self.X = X
        self.cols = cols
        self.stop_words = stop_words
        self.clean_vec = np.vectorize(self.clean)

    def unique_names(self, names, col, drugs, conditions):

        if col == "drugName":
            drugs.update(set(names))
        elif col == "condition":
            conditions.update(set(names))

    def clean(self, textinp, col):
        # makes text lowercase, removes punctuations and digits

        text = textinp.lower()
        text = text.translate(str.maketrans(
            string.punctuation, ' '*len(string.punctuation)))
        text = text.translate(str.maketrans(
            string.digits, ' '*len(string.digits)))

        word_tokens = word_tokenize(text)
        clean_text = []

        for word in word_tokens:
            if word not in self.stop_words:
                clean_text.append(word)

        self.unique_names(clean_text, col)

        return " ".join(clean_text)

    def apply_clean_to_cell(self):

        X_res = (self.X).copy()

        for col in self.cols:
            X_res.loc[:, col] = self.clean_vec((self.X).loc[:, col], col)

        return X_res
