<img width="1536" height="1024" alt="Autoencoder and Classifier for Digit Recognition" src="https://github.com/user-attachments/assets/669759b0-9f00-4815-853f-a79808c52988" />

Autoencoder + Classifier on MNIST
Unsupervised Feature Learning + Supervised Classification

This project demonstrates a complete 3-step Deep Learning pipeline using MNIST digits:

Autoencoder → Learn compressed representation

Encoder → Extract features

Classifier → Predict digit using extracted features

Project Structure
Autoencoder
 
 ├── Encoder          → Dimensionality reduction (784 → 64)

 ├── Decoder          → Reconstruct image (64 → 784)

 └── Classifier       → Digit classification using encoded features


Using MNIST handwritten digits dataset:

60,000 training images

10,000 test images

Grayscale (28×28) images flattened to 784-dim vectors


Autoencoder

Input: 784  
Encoder: Dense(64, relu)  
Decoder: Dense(784, sigmoid)  
Loss: Mean Squared Error (MSE)

Classifier

Input: Encoded features (64)  
Dense(10, softmax)  
Loss: Categorical Crossentropy

Code Overview

Load & Normalize Data
MNIST images reshaped to (784,)
Pixel values scaled to 0–1

Build Autoencoder

Encoder learns compressed 64-D representation
Decoder reconstructs original image

Train Autoencoder

Train for 5 epochs
Loss: MSE

Train Classifier

Uses encoder output as input
Trains for 5 epochs
Tracks loss + accuracy

Evaluate

Shows original vs reconstructed images
Prints accuracy of classifier
Predicts first 10 digits

Results
Autoencoder

Successfully reconstructs MNIST digits.
Bottom images are reconstructed from compressed 64-D features.

Classifier

Provides good accuracy using compressed embeddings.
Accuracy improves with more training epochs.

Visualization (Output)

First row → Original MNIST digits

Second row → Autoencoder reconstructed digits

<img width="949" height="337" alt="digit_recogntion" src="https://github.com/user-attachments/assets/20181c25-6915-498d-b127-9be726cacf3b" />

