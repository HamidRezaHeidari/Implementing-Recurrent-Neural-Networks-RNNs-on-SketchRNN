######### if face with Hardware Warning then uncomment these:
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
#########

import tensorflow as tf
import keras
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# Preprocess the drawings
def preprocess_drawings(drawings, max_len):
    processed_drawings = []
    for drawing in drawings:
        flat_drawing = [point for stroke in drawing for point in stroke]
        processed_drawings.append(flat_drawing)

    padded_drawings = pad_sequences(processed_drawings,maxlen=max_len)
    return padded_drawings

# Define parameters
num_classes = 5
max_len = 750    # Max Length of drawing datas

# Load downloaded Dataset and separate samples of 5 class
df = pd.read_parquet('0000.parquet')      #download from www.huggingface.co/datasets/quickdraw/tree/refs%2Fconvert%2Fparquet/sketch_rnn/partial-train
group = df.groupby("word")
df = pd.concat([group.get_group(0), group.get_group(1)
      , group.get_group(2), group.get_group(3), group.get_group(4)])

# Print number of datas on each class
print(df["word"].value_counts())

# Split test and train
train, test = train_test_split(df, test_size=0.2, stratify=df["word"], random_state=657)

# Train data
train_drawings = train["drawing"]
train_labels = train["word"]
X_train = preprocess_drawings(train_drawings, max_len)
y_train = to_categorical(train_labels, num_classes=num_classes)


# Test data
test_drawings = test["drawing"]
test_labels = test["word"]
X_test = preprocess_drawings(test_drawings, max_len)
y_test = to_categorical(test_labels, num_classes=num_classes)

# Define Hyperparameters
num_samples = len(train_labels)
vocab_size = num_samples
embedding_dim = 128
batch_size = 2500
epochs = 25


# Define the RNN model with an Embedding layer
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim),
    tf.keras.layers.SimpleRNN(64, activation='relu', return_sequences=True),
    tf.keras.layers.SimpleRNN(64, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# define Model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

# Train the model
model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.15)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print('Test accuracy:', accuracy ,"\n", 'Test loss:', loss)
print("finish")
