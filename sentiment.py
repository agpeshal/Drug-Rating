import pandas as pd
import numpy as np
from utils import myrow_concatenator
from textblob import TextBlob
import matplotlib.pyplot as plt


def calculate_polarity_subjectivity(X, textcolumns):
    """
    returns a matrix containing polarity and subjectivity of the entries in textcolumns
    """
    reviews = X.loc[:, textcolumns].values
    # add all texts for one sample to one text
    if reviews.size > reviews.shape[0]:
        reviews = np.apply_along_axis(myrow_concatenator, 1, reviews)
    n = reviews.shape[0]
    polarities = np.zeros(n)
    subjectivities = np.zeros(reviews.shape[0])

    for idx, review in enumerate(reviews):
        blob = TextBlob(review)
        polarities[idx], subjectivities[idx] = blob.sentiment
        if idx % 1000 == 0:
            print(f"{idx}/{n} = {np.round(100*idx/n,1)}% done")

    return pd.DataFrame({'polarity': polarities, 'subjectivity': subjectivities})

