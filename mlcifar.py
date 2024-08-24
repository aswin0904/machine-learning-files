import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras import layers ,models
from tensorflow.keras import utils
from tensorflow.keras.optimizers import Adam
(x_train,y_train),(x_test,y_test)=tf.keras.datasets.cifar10.load_data()
cls=['Airplanes','Automobile','Bird','Cat','Deer','Dog','Frog','Horse','Ship','Truck']
'''plt.figure(figsize=(10,10))
for i in range(200):
  plt.subplot(20,20,i+1)
  plt.xticks([])
  plt.yticks([])
  plt.grid(False)
  plt.imshow(x_train[i])
  plt.xlabel(cls[y_train[i][0]])
plt.show()'''
x_train,x_test=x_train/255.0,x_test/255.0
y_train=tf.keras.utils.to_categorical(y_train,10) # Use tf.keras.utils instead of tf.utils
y_test=tf.keras.utils.to_categorical(y_test,10)
model=models.Sequential([
    layers.Conv2D(16,(3,3),activation='relu',input_shape=(32,32,3)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(32,(3,3),activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64,(3,3),activation='relu'),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(64,activation='relu'),
    
    
    layers.Dense(10,activation='softmax')
])
# For integer labels (0, 1, 2, ...):
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# For one-hot encoded labels:
# model.compile(optimizer='adam',
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])

history=model.fit(x_train,y_train,epochs=10,batch_size=128,validation_data=(x_test,y_test))
model.evaluate(x_test,y_test)