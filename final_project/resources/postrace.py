from pymongo import MongoClient
import pandas as pd
import numpy as np
import re
from nltk.tokenize import WordPunctTokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import load_model
from keras.optimizers import TFOptimizer
from keras import models
from keras import layers
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold

NUM_WORDS = 30000
MAX_LENGTH = 250
MONGO_URL = ''

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


train_df = pd.read_csv('resources/train.csv', names=['rating', 'title', 'review'])
test_df = pd.read_csv('resources/test.csv', names=['rating', 'title', 'review'])
train_X = clean_reviews(train_df['review'])['review']
train_Y = train_df['rating']
test_X = clean_reviews(test_df['review'])['review']
test_Y = test_df['rating']
all_X = pd.concat([train_X, test_X])

tokenizer = Tokenizer(num_words=NUM_WORDS)
tokenizer.fit_on_texts(all_X)

client = MongoClient(MONGO_URL)
db = client.calvinpostrace

postrace_df = pd.concat([pd.DataFrame(db.archives.find()), pd.DataFrame(db.races.find())], ignore_index=True)


def scale_feature(feature):
    if feature < 5:
        return 1
    elif 5 <= feature <= 6:
        return 2
    elif feature == 7:
        return 3
    elif feature == 8:
        return 4
    elif feature > 8:
        return 5
    else:
        raise ValueError

analysis_list = []
attitude_list = []
effort_list = []
for index, row in postrace_df.iterrows():
    try:
        attitude = scale_feature(round(float(row.attitude)))
        effort = scale_feature(round(float(row.effort)))
        attitude_list.append(attitude)
        effort_list.append(effort)
    except ValueError:
        continue

    analysis_parts = [row.thoughts, row.positives, row.goal, row.turnpoint]
    analysis_parts = [x for x in analysis_parts if str(x) != "nan"]
    analysis_list.append(' '.join(analysis_parts))

X = clean_reviews(analysis_list)['review']
Y1 = np.array(attitude_list)
Y2 = np.array(effort_list)

X = preprocess_reviews(X, tokenizer)

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=5)
attitude_scores = []
effort_scores = []
attitude_confusion_matrices = []
effort_confusion_matrices = []
count = 1
for train, test in kfold.split(X, Y1, Y2):
    train_Y1 = np.delete(to_categorical(Y1[train]), 0, axis=1)
    test_Y1 = np.delete(to_categorical(Y1[test]), 0, axis=1)
    train_Y2 = np.delete(to_categorical(Y2[train]), 0, axis=1)
    test_Y2 = np.delete(to_categorical(Y2[test]), 0, axis=1)

    amzn_model = load_model('resources/amazon.h5')
    amzn_model.layers[0].trainable = False
    input1 = layers.Input(shape=(MAX_LENGTH,))
    x = amzn_model.layers[0](input1)
    x = amzn_model.layers[1](x)
    out1 = amzn_model.layers[2](x)
    out2 = amzn_model.layers[2](x)
    postrace_model = models.Model(inputs=input1, outputs=[out1, out2])

    if count == 1:
        postrace_model.summary()

    postrace_model.compile(optimizer=TFOptimizer(tf.optimizers.Adam()),
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

    history = postrace_model.fit(X[train],
                                 [train_Y1, train_Y2],
                                 epochs=150,
                                 batch_size=32,
                                 validation_data=(X[test], [test_Y1, test_Y2]))
    score = postrace_model.evaluate(X[test], [test_Y1, test_Y2], verbose=0)
    print("Run %d attitude accuracy: %.2f%%" % (count, score[3] * 100))
    attitude_scores.append(score[3] * 100)
    print("Run %d effort accuracy: %.2f%%" % (count, score[4] * 100))
    effort_scores.append(score[4] * 100)

    pred_Y = np.array(postrace_model.predict(X[test]))
    pred_Y1 = [np.argmax(x) for x in pred_Y[0]]
    pred_Y2 = [np.argmax(x) for x in pred_Y[1]]
    true_Y1 = [np.argmax(x) for x in test_Y1]
    true_Y2 = [np.argmax(x) for x in test_Y2]
    attitude_confusion_matrices.append(tf.math.confusion_matrix(labels=true_Y1, predictions=pred_Y1))
    effort_confusion_matrices.append(tf.math.confusion_matrix(labels=true_Y2, predictions=pred_Y2))

    count += 1

print("attitude accuracy: %.2f%% (+/- %.2f%%)" % (np.mean(attitude_scores), np.std(attitude_scores)))
print("effort accuracy: %.2f%% (+/- %.2f%%)" % (np.mean(effort_scores), np.std(effort_scores)))

attitude_confusion_matrix = np.sum(np.array(attitude_confusion_matrices), axis=0)
attitude_confusion_matrix = np.around(attitude_confusion_matrix.astype('float') / attitude_confusion_matrix.sum(axis=1)[:, np.newaxis], decimals=2)
attitude_confusion_matrix = pd.DataFrame(attitude_confusion_matrix, index=[1, 2, 3, 4, 5], columns=[1, 2, 3, 4, 5])

plt.clf()
figure = plt.figure(figsize=(8, 8))
sns.heatmap(attitude_confusion_matrix, annot=True, cmap=plt.cm.Blues)
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

effort_confusion_matrix = np.sum(np.array(effort_confusion_matrices), axis=0)
effort_confusion_matrix = np.around(effort_confusion_matrix.astype('float') / effort_confusion_matrix.sum(axis=1)[:, np.newaxis], decimals=2)
effort_confusion_matrix = pd.DataFrame(effort_confusion_matrix, index=[1, 2, 3, 4, 5], columns=[1, 2, 3, 4, 5])

plt.clf()
figure = plt.figure(figsize=(8, 8))
sns.heatmap(effort_confusion_matrix, annot=True, cmap=plt.cm.Blues)
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
