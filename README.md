## 🧠 Sequence Learning in Action: Using RNNs for Hand-Drawn Sketch Recognition
Recurrent Neural Networks (RNNs) are powerful architectures built to handle temporal and sequential data — where each new input depends on what came before.
They’ve become essential in applications like speech recognition, handwriting analysis, and time-series prediction.
---
##🎨 In this project, I explored an RNN-based approach using the SketchRNN dataset — a structured dataset of vectorized sketches from Google’s QuickDraw project.
Each sketch is represented as a sequence of pen strokes, making it an ideal testbed for sequence modeling.
---
## ⚙️ Model Overview

The network includes:
-An Embedding layer to convert strokes into dense vector representations
-Two SimpleRNN layers with ReLU activation for sequential feature extraction
-A Dense softmax output layer for classification across 5 sketch categories

## 🧩 Key Insights

1️⃣The model learns to distinguish sketches purely from temporal stroke data, without relying on image pixels.
2️⃣Despite the simplicity of the RNN architecture, it achieved strong accuracy and generalization on unseen sketches.
3️⃣Demonstrates how sequence-based neural networks can understand human drawing behavior as dynamic, time-dependent data.
<br/>
✨Star if you find it useful

