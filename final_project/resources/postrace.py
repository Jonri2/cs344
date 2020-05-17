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
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold

NUM_WORDS = 100000
MAX_LENGTH = 440
VECTOR_DIM = 200
LEARNING_RATE = 0.00005
DEBUG = True


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

train_df = pd.read_csv('full_train.csv', names=['rating', 'title', 'review'])
test_df = pd.read_csv('test.csv')
train_X = clean_reviews(train_df['review'])['review']
train_Y = train_df['rating']
test_X = clean_reviews(test_df['review'])['review']
test_Y = test_df['rating']
all_X = pd.concat([train_X, test_X])

tokenizer = Tokenizer(num_words=NUM_WORDS)
tokenizer.fit_on_texts(all_X)

client = MongoClient('mongodb://admin:track1@ds141815.mlab.com:41815/calvinpostrace')
db = client.calvinpostrace

postrace_df = pd.concat([pd.DataFrame(db.archives.find()), pd.DataFrame(db.races.find())], ignore_index=True)

analysis_list = []
rating_list = []
for index, row in postrace_df.iterrows():
    try:
        rating = round((float(row.attitude) + float(row.effort))/2)
        if rating < 6:
            rating = 1
        elif rating == 6 or rating == 7:
            rating = 2
        elif rating == 8:
            rating = 3
        elif rating == 9:
            rating = 4
        elif rating > 9:
            rating = 5
        else:
            continue
        rating_list.append(rating)
    except ValueError:
        continue

    analysis_parts = [row.thoughts, row.positives, row.goal, row.turnpoint]
    analysis_parts = [x for x in analysis_parts if str(x) != "nan"]
    analysis_list.append(' '.join(analysis_parts))

X = clean_reviews(analysis_list)['review']
Y = np.array(rating_list)

X = preprocess_reviews(X, tokenizer)

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=5)
scores = []
confusion_matrices = []
for train, test in kfold.split(X, Y):
    train_Y = np.delete(to_categorical(Y[train]), 0, axis=1)
    test_Y = np.delete(to_categorical(Y[test]), 0, axis=1)
    model = load_model('amazon.h5')

    model.compile(optimizer=TFOptimizer(tf.optimizers.Adam(LEARNING_RATE)),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(X[train],
                        train_Y,
                        epochs=30,
                        batch_size=8)
    score = model.evaluate(X[test], test_Y, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], score[1] * 100))
    scores.append(score[1] * 100)

    pred_Y = model.predict_classes(X[test])
    true_Y = [np.argmax(x) for x in test_Y]
    confusion_matrices.append(tf.math.confusion_matrix(labels=true_Y, predictions=pred_Y).numpy())

print("%.2f%% (+/- %.2f%%)" % (np.mean(scores), np.std(scores)))

# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']
# loss = history.history['loss']
# val_loss = history.history['val_loss']
#
# epochs = range(1, len(acc) + 1)
#
# # "bo" is for "blue dot"
# plt.plot(epochs, loss, 'bo', label='Training loss')
# # b is for "solid blue line"
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
#
# plt.show()

full_confusion_matrix = np.sum(np.array(confusion_matrices), axis=0)
confusion_matrix = np.around(full_confusion_matrix.astype('float') / full_confusion_matrix.sum(axis=1)[:, np.newaxis], decimals=2)
full_confusion_matrix = pd.DataFrame(full_confusion_matrix, index=[1, 2, 3, 4, 5], columns=[1, 2, 3, 4, 5])

plt.clf()
figure = plt.figure(figsize=(8, 8))
sns.heatmap(full_confusion_matrix, annot=True, cmap=plt.cm.Blues)
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
