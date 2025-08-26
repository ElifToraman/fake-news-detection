import time

import pandas as pd
import numpy as np
from keras import Input
from nltk.tokenize import word_tokenize
from gensim.models import KeyedVectors
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


# Load Dataset
df = pd.read_csv("data/fake_or_real_news.csv")
df = df[["text", "label"]]
df.dropna(inplace=True)

# Load Pretrained Word2Vec
print("Loading Word2Vec model...")
w2v_path = "data/GoogleNews-vectors-negative300.bin"
w2v_model = KeyedVectors.load_word2vec_format(w2v_path, binary=True, limit=100000)
print("Word2Vec model loaded.")

# Preprocessing & Embeddings
df['tokens'] = df['text'].apply(lambda x: word_tokenize(x.lower()))

def document_vector(tokens):
    vectors = [w2v_model[word] for word in tokens if word in w2v_model]
    return np.mean(vectors, axis=0) if vectors else np.zeros(w2v_model.vector_size)

df['embedding'] = df['tokens'].apply(document_vector)

X = np.vstack(df['embedding'].values)
y = df['label'].map({'FAKE': 0, 'REAL': 1})

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=int(time.time()))

# Build Neural Network
model = Sequential()
model.add(Input(shape=(X_train.shape[1],)))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Train Model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate Model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\n Test Accuracy: {accuracy:.4f}")

y_pred = (model.predict(X_test) > 0.5).astype("int32")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["FAKE", "REAL"], yticklabels=["FAKE", "REAL"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# Plot Accuracy and Loss
def plot_training_history(history):
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

plot_training_history(history)

# Try Your Own Article
def predict_news(text):
    tokens = word_tokenize(text.lower())
    vector = document_vector(tokens).reshape(1, -1)
    prediction = model.predict(vector)[0][0]
    label = "REAL" if prediction >= 0.5 else "FAKE"
    confidence = prediction if prediction >= 0.5 else 1 - prediction
    print(f"\n Your News: {text}")
    print(f" Prediction: {label} ({confidence * 100:.2f}% confidence)")

while True:
    print("\n Try your own news article below:")
    sample_text = input("Type a news article to test: ")
    predict_news(sample_text)

