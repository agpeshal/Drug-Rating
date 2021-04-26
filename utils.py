import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error, f1_score, accuracy_score


def labelencode(y):
    # encoding based on task description
    y_res = np.zeros(y.shape[0])
    y_res[y >= 4] = 1
    y_res[y >= 7] = 2

    return y_res


def myrow_concatenator(r):
    # concatenates texts from a row of a matrix
    totalstr = ''
    for elem in r:
        if elem == elem:  # doesn't hold for nans
            totalstr = totalstr + " " + elem
    return totalstr


def fit_randomforest(xtrain, ytrain, xval, yval, depths, training_scores=False):
    if training_scores:
        xval = xtrain
        yval = ytrain

    for d in depths:

        clf = RandomForestClassifier(
            n_estimators=100, max_depth=d, random_state=42)
        clf.fit(xtrain, ytrain)
        pred_val = clf.predict(xval)

        f1 = f1_score(yval, pred_val, average='macro')
        acc = accuracy_score(yval, pred_val)

        print("For Depth: {}, F1 score: {:.2f}, Accuracy: {:.2f}".format(d, f1, acc))


def fit_rf_regression(xtrain, ytrain, xval, yval, depths,
                      n_estimators=10, random_state=42):
    # Random Forest

    for d in depths:
        clf = RandomForestRegressor(
            n_estimators=n_estimators, max_depth=d, random_state=random_state)
        clf.fit(xtrain, ytrain)
        pred_val = clf.predict(xval)
        mse = mean_squared_error(yval, pred_val)

        print("For Depth: {}, MSE: {:.2f}".format(d, mse))


def plot_polarity_subj(subjectivity, polarity, y_set):
    colors = ["red" if y == 0 else "green"
              if y == 2 else "orange" for y in y_set]
    plt.scatter(subjectivity, polarity, c=colors, s=10)
    plt.xlabel("Subjectivity")
    plt.ylabel("Polarity")
    plt.title(
        "Sentiment (green = positive, orange = neutral, red = negative) by Polarity and Subjectivity")


def transform_doc2vec(text, model):
    text = text.split()
    return model.infer_vector(text)


def add_doc2vec_columns(df, textcolumns, model):
    X = df.copy().drop(columns=textcolumns)
    for col in textcolumns:
        embeddings = df[col].apply(transform_doc2vec, model=model)
        expanded = embeddings.apply(pd.Series)
        expanded = expanded.add_prefix(col+'_')
        X = pd.concat([X, expanded], axis=1)
    return X


def diag2vec(sentence, w2v_model):
    # aggregates sentences by taking the mean of the word2vec embedding vectors
    if pd.isna(sentence):
        return np.zeros((200,))
    words = [w for w in sentence.split() if w in w2v_model.vocab]
    if len(words) == 0:
        return np.zeros((200,))
    emb = np.zeros((len(words), 200))
    for i, w in enumerate(words):
        emb[i, :] = w2v_model[w]
    return emb.mean(axis=0)


def pmc_diag2vec(sentence, pmc_map):
    return diag2vec(sentence, pmc_map)


def add_diag2vec_columns(df, textcolumns):
    X = df.copy().drop(columns=textcolumns)
    for col in textcolumns:
        embeddings = df[col].apply(pmc_diag2vec)
        expanded = embeddings.apply(pd.Series)
        expanded = expanded.add_prefix(col+'_')
        X = pd.concat([X, expanded], axis=1)
    return X

