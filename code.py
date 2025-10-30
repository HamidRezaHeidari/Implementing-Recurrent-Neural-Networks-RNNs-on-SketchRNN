########## Hardware Warning Fix ##########
# Uncomment the following lines if you face TensorFlow hardware warnings
# import os
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
##########################################

# ===============================
#   Import Required Libraries
# ===============================
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ===============================
#   Data Preprocessing Function
# ===============================
def preprocess_drawings(drawings, max_len):
    """
    Preprocess and pad raw QuickDraw sketches for model training.

    Parameters
    ----------
    drawings : list
        List of drawing data, where each drawing consists of strokes (sequences of points).
    max_len : int
        Maximum length of flattened and padded sequences.

    Returns
    -------
    numpy.ndarray
        Padded and flattened sequence representation of drawings.
    """
    processed_drawings = []

    # Flatten each drawing (list of strokes → list of points)
    for drawing in drawings:
        flat_drawing = [point for stroke in drawing for point in stroke]
        processed_drawings.append(flat_drawing)

    # Pad sequences to a fixed length
    padded_drawings = pad_sequences(processed_drawings, maxlen=max_len)
    return padded_drawings


# ===============================
#   Define Parameters
# ===============================
num_classes = 5        # Number of target classes
max_len = 750          # Maximum sequence length per sample
batch_size = 2500      # Batch size for training
epochs = 25            # Number of epochs for training
embedding_dim = 128    # Dimension of embedding vectors
random_state = 657     # Random state for reproducibility


# ===============================
#   Load and Prepare Dataset
# ===============================
"""
Dataset Source:
https://huggingface.co/datasets/quickdraw/tree/refs%2Fconvert%2Fparquet/sketch_rnn/partial-train

The dataset should be in Parquet format (e.g., '0000.parquet') and
contain at least the following columns: 'drawing' and 'word'.
"""

df = pd.read_parquet('0000.parquet')

# Select only 5 classes for this experiment
group = df.groupby("word")
df = pd.concat([
    group.get_group(0),
    group.get_group(1),
    group.get_group(2),
    group.get_group(3),
    group.get_group(4)
])

# Print the number of samples in each class
print("Number of samples per class:\n", df["word"].value_counts(), "\n")

# ===============================
#   Split Data into Train/Test
# ===============================
train_df, test_df = train_test_split(
    df, test_size=0.2, stratify=df["word"], random_state=random_state
)

# Prepare training data
X_train = preprocess_drawings(train_df["drawing"], max_len)
y_train = to_categorical(train_df["word"], num_classes=num_classes)

# Prepare testing data
X_test = preprocess_drawings(test_df["drawing"], max_len)
y_test = to_categorical(test_df["word"], num_classes=num_classes)


# ===============================
#   Define RNN Model
# ===============================
"""
A simple Recurrent Neural Network (RNN) is used to classify QuickDraw sketches.

Architecture:
- Embedding layer: Learns vector representations for input sequences
- Two SimpleRNN layers: Extract temporal dependencies in strokes
- Dense layer (Softmax): Outputs class probabilities
"""

vocab_size = len(train_df)  # Use number of training samples as vocabulary size

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim),
    tf.keras.layers.SimpleRNN(64, activation='relu', return_sequences=True),
    tf.keras.layers.SimpleRNN(64, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ===============================
#   Train the Model
# ===============================
history = model.fit(
    X_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.15,
    verbose=1
)

# ===============================
#   Evaluate Model Performance
# ===============================
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test Loss: {loss:.4f}")
print("Training completed successfully.")

# ===============================
#   Optional: Plot Training History
# ===============================
plt.figure(figsize=(10, 4))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy During Training')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
