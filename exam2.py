from tensorflow.keras import layers
from tensorflow.keras import models

from keras.datasets import mnist
from keras.utils import to_categorical

(train_images, train_labels), (test_images, test_labels) =\
 mnist.load_data()
 
train_images = train_images.reshape((60000, 28, 28, 1)) 
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

model = models.Sequential()
model.add(layers.Conv2D(8,(3, 3),
                        padding='same',
                        activation='relu',input_shape=(28,28, 1)))

model.add(layers.MaxPooling2D((2, 2)))

model = models.Sequential()
model.add(layers.Conv2D(8,(5, 5),
                        padding='same',
                        activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())

model.add(layers.Dense(16, activation='relu'))

model.add(layers.Dense(10, activation='softmax'))



from keras import losses
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy', #losses.sparse_categorical_crossentropy
              metrics=['acc'])

history = model.fit(train_images, 
          train_labels, 
          epochs=5, 
          batch_size=64,
          validation_split=0.1)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print("test_acc=",test_acc)




model.save('mnist_conv_model')


import matplotlib.pyplot as plt

plt.figure(2)
acc = history.history['acc']
val_acc = history.history['val_acc']
plt.plot( acc, 'ro', label='Training acc')
plt.plot( val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('acc')
plt.legend()