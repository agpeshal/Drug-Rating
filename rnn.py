import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from gensim.models import KeyedVectors
from keras.models import Model
from keras.layers import Input, Dense, LSTM, Embedding
from keras import optimizers
from keras.preprocessing.text import Tokenizer
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from preprocessing import Preprocessing
from nltk.corpus import stopwords

MAX_WORDS = 100
EMBDEDDING_SIZE = 300


def get_model(vocab_size, embedding_matrix):
    inp = Input(shape=(100,))
    embedding = Embedding(vocab_size, EMBDEDDING_SIZE, input_length=MAX_WORDS, weights=[
                          embedding_matrix], trainable=False)(inp)
    lstm = LSTM(100)(embedding)
    output = Dense(3, activation='softmax')(lstm)
    model = Model(inputs=inp, outputs=output)

    return model


def get_embedding(tk):
    vocab = tk.word_counts.keys()
    vocab_size = len(vocab) + 1
    non_zero = 0
    embedding_matrix = np.zeros((vocab_size, 300))
    mapping = KeyedVectors.load_word2vec_format(
        'GoogleNews-vectors-negative300.bin', binary=True)

    for idx, word in enumerate(vocab):

        try:
            embedding_vec = mapping[word]
            non_zero += 1
        except:
            embedding_vec = None

        if embedding_vec is not None:
            # Spent hours to debug this "+1"
            embedding_matrix[idx + 1] = embedding_vec

    return embedding_matrix


def train(tk, X_train, y_train, epochs=50, lr=0.0001, batch=256, val_size=0.2):
    vocab = tk.word_counts.keys()
    vocab_size = len(vocab) + 1
    embedding_matrix = get_embedding(tk)

    model = get_model(vocab_size=vocab_size, embedding_matrix=embedding_matrix)
    adam = optimizers.Adam(lr=lr)
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)
    chk_pt = ModelCheckpoint(
        "best.model.hdf5", save_best_only=True, monitor='val_loss')
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam, metrics=['accuracy'])
    model.summary()

    y_train_ohe = pd.get_dummies(y_train)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train_ohe, test_size=val_size, random_state=42)
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              batch_size=batch, epochs=epochs, callbacks=[reduce_lr, chk_pt])

    return model


def get_data():
    train = pd.read_csv('drugsComTrain_raw.tsv', sep='\t',
                        header=0, engine='python', error_bad_lines=False)
    train = train.rename(columns={'Unnamed: 0': 'Id'})
    train = train.set_index('Id')
    X_train = train.loc[:, train.columns != "rating"]
    y_train = train.loc[:, "rating"]

    test = pd.read_csv('drugsComTest_raw.tsv', sep='\t',
                       header=0, engine='python', error_bad_lines=False)
    test = test.rename(columns={'Unnamed: 0': 'Id'})
    test = test.set_index('Id')
    X_test = test.loc[:, test.columns != "rating"]
    y_test = test.loc[:, "rating"]

    stop_words = set(stopwords.words('english'))
    text_cols = ["drugName", "condition", "review"]

    X_train_clean = Preprocessing(
        X_train, text_cols, stop_words).apply_clean_to_cell()
    X_test_clean = Preprocessing(
        X_test, text_cols, stop_words).apply_clean_to_cell()

    return X_train_clean, y_train, X_test_clean, y_test


if __name__ == "__main__":

    X_train, y_train, X_test, y_test = get_data()
    X = pd.concat([X_train, X_test], axis=0).values
    tk = Tokenizer(lower=True)
    tk.fit_on_texts(X)

    model = train(tk, X_train, y_train)
    score = model.evaluate(X_test, pd.get_dummies(y_test))

    print(f"Test loss {score[0]}, Test accuracy: {score[1]}")
