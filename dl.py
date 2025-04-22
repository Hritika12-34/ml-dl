#load libraries
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding,Dense,Flatten
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt

#load dataset
(x_train, y_train),(x_test, y_test)= imdb.load_data(num_words=10000)

#pad-sequences
x_train= pad_sequences(x_train,maxlen=200)
x_test= pad_sequences(x_test,maxlen=200)

#build
model= Sequential([
    Embedding(input_dim=10000,output_dim=32,input_length=200),
    Flatten(),
    Dense(64,activation='relu'),
    Dense(1,activation='sigmoid')
])

#compile
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

#train
history= model.fit(
    x_train,y_train,
    epochs=5,
    batch_size=512,
    validation_split=0.2
)


#evaluate
loss,acc=model.evaluate(x_test,y_test)
print(f"Test Accuracy: {acc:.4f}")

#visualize

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label="Train_Accuracy")
plt.plot(history.history['val_accuracy'], label="Val_Accuracy")
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()


plt.plot(history.history['loss'], label="Loss")
plt.plot(history.history['val_loss'], label="Val_loss")
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
