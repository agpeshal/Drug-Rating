import pandas as pd
import numpy as np
import spacy
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from utils import myrow_concatenator

nlpspacy = spacy.load('en', disable=['parser', 'ner'])
ps = PorterStemmer()


def lemmatize_text(text):
    # spacys lemmatizer superior to nltk or textblobs because it correctly handles verbs and pronouns
    # without needing the appropriate POS tag as input
    return " ".join([token.lemma_ for token in nlpspacy(str(text))])


def stem_text(text):
    return " ".join([ps.stem(word) for word in word_tokenize(text)])


def add_per_cell_text_features(X, textcolumns, toapplyfunc, colname):
    to_transform = X.loc[:, textcolumns].values
    # add all texts for one sample to one text
    if to_transform.size > to_transform.shape[0]:
        to_transform = np.apply_along_axis(myrow_concatenator, 1, to_transform)
    n = to_transform.shape[0]
    result_texts = ['']*n

    for idx, text in enumerate(to_transform):
        result_texts[idx] = toapplyfunc(text)
        if idx % 1000 == 0:
            print(f"{idx}/{n} = {np.round(100*idx/n,1)}% done")
    transformed_df = pd.DataFrame({colname: result_texts}, index=X.index)
    X_res = X.drop(textcolumns, axis=1)
    X_res = pd.concat([X_res, transformed_df], axis=1)
    return X_res


def add_lemmatization_features(X, textcolumns):
    return add_per_cell_text_features(X, textcolumns, lemmatize_text, 'lemmatized_texts')


def add_stemming_features(X, textcolumns):
    return add_per_cell_text_features(X, textcolumns, stem_text, 'stemmed_texts')
