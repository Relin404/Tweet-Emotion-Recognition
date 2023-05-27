from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense
from tensorflow.keras.models import Sequential
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import datasets
import random
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

dataset = datasets.load_dataset("dair-ai/emotion")


def show_history(h):
    epochs_trained = len(h.history['loss'])
    plt.figure(figsize=(16, 6))

    plt.subplot(1, 2, 1)
    plt.plot(range(0, epochs_trained), h.history.get(
        'accuracy'), label='Training')
    plt.plot(range(0, epochs_trained), h.history.get(
        'val_accuracy'), label='Validation')
    plt.ylim([0., 1.])
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(0, epochs_trained), h.history.get('loss'), label='Training')
    plt.plot(range(0, epochs_trained), h.history.get(
        'val_loss'), label='Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def show_confusion_matrix(y_true, y_pred, classes):
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred, normalize='true')
    plt.figure(figsize=(8, 8))
    sp = plt.subplot(1, 1, 1)
    ctx = sp.matshow(cm)
    plt.xticks(list(range(0, 6)), labels=classes)
    plt.yticks(list(range(0, 6)), labels=classes)
    plt.colorbar(ctx)
    plt.show()


# Save the three subsets or splits into lists of dictionaries
testSet = dataset["test"]
trainSet = dataset["train"]
validationSet = dataset["validation"]


def getTweet(data):
    """
    Extract tweets and labels from a given list of dictionaries
    """
    tweets = [x["text"] for x in data]
    labels = [x["label"] for x in data]
    return tweets, labels


trainTweets, trainLabels = getTweet(trainSet)

tokenizer = Tokenizer(num_words=10000, oov_token="<UNK>")
tokenizer.fit_on_texts(trainTweets)

lengths = [len(t.split(" ")) for t in trainTweets]
plt.hist(lengths, bins=len(set(lengths)))

maxLen = 50


def getSequences(tokenizer, tweets):
    sequences = tokenizer.texts_to_sequences(tweets)
    paddedSequences = pad_sequences(
        sequences, truncating="post", padding="post", maxlen=maxLen
    )

    return paddedSequences


paddedTrainSequences = getSequences(tokenizer, trainTweets)


plt.hist(trainLabels, bins=11)
plt.show()

names = dataset["train"].features["label"].names
classToIndex = dict((c, i) for i, c in enumerate(names))
indexToClass = dict((i, c) for c, i in classToIndex.items())


model = Sequential([
    Embedding(10000, 16, input_length=maxLen),
    Bidirectional(LSTM(20, return_sequences=True)),
    Bidirectional(LSTM(20)),
    Dense(6, activation="softmax")
])

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

model.summary()

validationTweets, validationLabels = getTweet(validationSet)
validationSequences = getSequences(tokenizer, validationTweets)
validationLabels = np.array(validationLabels)


h = model.fit(
    paddedTrainSequences,
    np.array(trainLabels),
    validation_data=(validationSequences, validationLabels),
    epochs=20,
    callbacks=[
        EarlyStopping(monitor="val_accuracy", patience=2)
    ]
)

show_history(h)

testTweets, testLabels = getTweet(testSet)
testSequences = getSequences(tokenizer, testTweets)
testLabels = np.array(testLabels)

_ = model.evaluate(testSequences, testLabels)

i = random.randint(0, len(testLabels) - 1)

print("Sentence:", testTweets[i])
print("Emotion:", indexToClass[testLabels[i]])

p = model.predict(np.expand_dims(testSequences[i], axis=0))[0]
predictedClass = indexToClass[np.argmax(p).astype("uint8")]

print("Predicted Emotion:", predictedClass)

predictions = np.argmax(model.predict(testSequences), axis=-1)
predictions.shape, testLabels.shape

show_confusion_matrix(testLabels, preds, list(classes))
