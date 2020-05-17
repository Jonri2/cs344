import pandas as pd
import numpy as np
import tensorflow as tf
from keras.optimizers import TFOptimizer
from nltk.tokenize import WordPunctTokenizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import models
from keras import layers
from keras.utils import to_categorical
from gensim.models.word2vec import Word2Vec
from gensim.models import KeyedVectors
import multiprocessing
from sklearn import utils
import re
import matplotlib.pyplot as plt
import seaborn as sns

NUM_WORDS = 100000
MAX_LENGTH = 440
VECTOR_DIM = 200
DEBUG = True


def reviews_to_word_list(list_reviews):
    tokenizer = WordPunctTokenizer()
    reviews = []
    for text in list_reviews:
        txt = tokenizer.tokenize(text)
        reviews.append(txt)
    return reviews


def create_embedding_matrix(tokenizer, df, load=True):
    if load:
        if DEBUG:
            print("Loading Word2Vecs...")
        model_cbow = KeyedVectors.load('model_cbow.word2vec')
        model_sg = KeyedVectors.load('model_sg.word2vec')
    else:
        processed_reviews = reviews_to_word_list(df)
        cores = multiprocessing.cpu_count()
        if DEBUG:
            print("Creating cbow Word2Vec...")
        model_cbow = Word2Vec(sg=0, size=VECTOR_DIM // 2, negative=5, window=2, min_count=2, workers=cores)
        if DEBUG:
            print("Building cbow Word2Vec...")
        model_cbow.build_vocab(processed_reviews)
        if DEBUG:
            print("Training cbow Word2Vec...")
        model_cbow.train(utils.shuffle(processed_reviews), total_examples=model_cbow.corpus_count, epochs=30)

        if DEBUG:
            print("Creating skipgram Word2Vec...")
        model_sg = Word2Vec(sg=1, size=VECTOR_DIM // 2, negative=5, window=2, min_count=2, workers=cores)
        if DEBUG:
            print("Building skipgram Word2Vec...")
        model_sg.build_vocab(processed_reviews)
        if DEBUG:
            print("Training skipgram Word2Vec...")
        model_sg.train(utils.shuffle(processed_reviews), total_examples=model_sg.corpus_count, epochs=30)

        if DEBUG:
            print("Saving Word2Vecs...")
        model_cbow.save('model_cbow.word2vec')
        model_sg.save('model_sg.word2vec')

    if DEBUG:
        print("Filling Embedding Indices...")
    embeddings_index = {}
    for w in model_cbow.wv.vocab.keys():
        embeddings_index[w] = np.append(model_cbow.wv[w], model_sg.wv[w])

    if DEBUG:
        print("Initializing Embedding matrix...")
    embedding_matrix = np.zeros((NUM_WORDS, VECTOR_DIM))
    if DEBUG:
        print("Creating Embedding matrix...")
    for word, i in tokenizer.word_index.items():
        if i >= NUM_WORDS:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


def preprocess_reviews(df, tokenizer):
    sequences = tokenizer.texts_to_sequences(df)
    return pad_sequences(sequences, maxlen=MAX_LENGTH)


def clean_reviews(df):
    tokenizer = WordPunctTokenizer()
    cleaned_reviews = []
    for review in df:
        letters_only = re.sub("[^a-zA-Z]", " ", review)
        lower_case = letters_only.lower()
        words = tokenizer.tokenize(lower_case)
        cleaned_reviews.append((" ".join(words)).strip())
    return pd.DataFrame(cleaned_reviews, columns=['review'])


if DEBUG:
    print("Loading training examples...")
train_df = pd.read_csv('full_train.csv', names=['rating', 'title', 'review'])
if DEBUG:
    print("Loading testing examples...")
test_df = pd.read_csv('test.csv')

if DEBUG:
    print("Cleaning training examples...")
train_X = clean_reviews(train_df['review'])['review']
train_Y = train_df['rating']
if DEBUG:
    print("Cleaning testing examples...")
test_X = clean_reviews(test_df['review'])['review']
test_Y = test_df['rating']
all_X = pd.concat([train_X, test_X])

if DEBUG:
    print("Creating tokenizer...")
tokenizer = Tokenizer(num_words=NUM_WORDS)
tokenizer.fit_on_texts(all_X)
if DEBUG:
    print("Tokenizing reviews...")
train_X = preprocess_reviews(train_X, tokenizer)
test_X = preprocess_reviews(test_X, tokenizer)
embedding_matrix = create_embedding_matrix(tokenizer, all_X, load=True)

# vectorize labels
train_Y = np.delete(to_categorical(train_Y), 0, axis=1)
test_Y = np.delete(to_categorical(test_Y), 0, axis=1)

val_X = train_X[900000:]
val_Y = train_Y[900000:]
train_X = train_X[:900000]
train_Y = train_Y[:900000]

model = models.Sequential()
model.add(layers.embeddings.Embedding(NUM_WORDS, VECTOR_DIM, input_length=MAX_LENGTH, weights=[embedding_matrix], trainable=False))
model.add(layers.Bidirectional(layers.LSTM(32)))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(5, activation='softmax'))
model.summary()

model.compile(optimizer=TFOptimizer(tf.optimizers.Adam()),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_X,
                    train_Y,
                    epochs=15,
                    batch_size=32,
                    validation_data=(val_X, val_Y))

model.save('amazon.h5')

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

pred_Y = model.predict_classes(val_X)
val_Y = [np.argmax(x) for x in val_Y]
confusion_matrix = tf.math.confusion_matrix(labels=val_Y, predictions=pred_Y).numpy()
confusion_matrix = np.around(confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis], decimals=2)
confusion_matrix = pd.DataFrame(confusion_matrix, index=[1, 2, 3, 4, 5], columns=[1, 2, 3, 4, 5])

plt.clf()
figure = plt.figure(figsize=(8, 8))
sns.heatmap(confusion_matrix, annot=True, cmap=plt.cm.Blues)
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
