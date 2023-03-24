# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import os

window_size = 20

covid_vector = []
covid_label = []
for path in os.listdir("covid"):
  if("csv" not in path):
    pass
  else:
    df = pd.read_csv(os.path.join("covid",path))
    covid_data = df["HROS"].values

    for i in range(len(covid_data)-window_size):
      covid_vector.append(covid_data[i:i+window_size])
      covid_label.append("Covid diagnosed")
train_test_split = int(len(covid_label)*0.7)
covid_vector = np.array(covid_vector)

shuffler = np.random.permutation(len(covid_label))
covid_vector = covid_vector[shuffler]
covid_label = np.array(covid_label)[shuffler]

covid_train_vector = covid_vector[0:train_test_split]
covid_test_vector = covid_vector[train_test_split:]

covid_train_label = covid_label[0:train_test_split]
covid_test_label = covid_label[train_test_split:]

print(np.shape(covid_vector))
print(np.shape(covid_train_vector))
print(np.shape(covid_test_vector))

covid_vector

healthy_vector = []
healthy_label = []
for path in os.listdir("healthy"):
  if("csv" not in path):
    pass
  else:

    df = pd.read_csv(os.path.join("healthy",path))
    healthy_data = df["HROS"].values

    for i in range(len(healthy_data)-window_size):
      healthy_vector.append(healthy_data[i:i+window_size])
      healthy_label.append("Healthy")
healthy_vector = np.array(healthy_vector)

train_test_split = int(len(healthy_label)*0.7)
shuffler = np.random.permutation(len(healthy_label))
healthy_vector = healthy_vector[shuffler]
healthy_label = np.array(healthy_label)[shuffler]

healthy_train_vector = healthy_vector[0:train_test_split]
healthy_test_vector = healthy_vector[train_test_split:]

healthy_train_label = healthy_label[0:train_test_split]
healthy_test_label = healthy_label[train_test_split:]

other_vector = []
other_label = []
for path in os.listdir("other"):
  if("csv" not in path):
    pass
  else:
    df = pd.read_csv(os.path.join("other",path))
    other_data = df["HROS"].values

    for i in range(len(other_data)-window_size):
      other_vector.append(other_data[i:i+window_size])
      other_label.append("Other")
other_vector = np.array(other_vector)

train_test_split = int(len(other_label)*0.7)
shuffler = np.random.permutation(len(other_label))
other_vector = other_vector[shuffler]
other_label = np.array(other_label)[shuffler]

other_train_vector = other_vector[0:train_test_split]
other_test_vector = other_vector[train_test_split:]

other_train_label = other_label[0:train_test_split]
other_test_label = other_label[train_test_split:]

print(np.shape(covid_train_vector))
print(np.shape(healthy_train_vector))
print(np.shape(other_train_vector))
print(np.shape(other_test_vector))
print(len(healthy_train_label))
print(len(other_train_label))

minimum = np.min([np.shape(covid_train_vector)[0], np.shape(healthy_train_vector)[0], np.shape(other_train_vector)[0]])

train_x = np.concatenate((covid_train_vector[:minimum-1], healthy_train_vector[:minimum-1], other_train_vector[:minimum-1]), axis=0)
train_y = np.concatenate((covid_train_label[:minimum-1], healthy_train_label[:minimum-1], other_train_label[:minimum-1]), axis=0)

test_x = np.concatenate((covid_test_vector, healthy_test_vector, other_test_vector), axis=0)
test_y = np.concatenate((covid_test_label, healthy_test_label, other_test_label), axis=0)

print(np.shape(train_x))
print(np.shape(train_y))

print(np.shape(test_x))
print(np.shape(test_y))

len(other_train_label)

shuffler = np.random.permutation(len(train_y))
train_x = train_x[shuffler]
train_y = train_y[shuffler]

np.mean(np.concatenate((train_x,test_x), axis = 0))

"""
train_x = train_x/72
test_x = test_x/72
"""



from keras import models
from keras import layers
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

dictt = {'Covid diagnosed': 0, 'Healthy': 1, 'Other': 2}

train_y = list((pd.Series(train_y)).map(dictt))
test_y = list((pd.Series(test_y)).map(dictt))

train_labels = to_categorical(train_y)
test_labels = to_categorical(test_y)

test_labels

train_x = train_x.reshape((len(train_x), window_size, 1))
test_x = test_x.reshape((len(test_x), window_size, 1))

"""
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_dim=window_size))
network.add(layers.Dropout(0.3))
network.add(layers.Dense(256, activation='relu'))
network.add(layers.Dropout(0.3))
network.add(layers.Dense(128, activation='relu'))
network.add(layers.Dropout(0.3))
network.add(layers.Dense(64, activation='relu'))
network.add(layers.Dropout(0.3))
network.add(layers.Dense(32, activation='relu'))
network.add(layers.Dropout(0.3))
network.add(layers.Dense(3, activation='softmax'))
"""

normalizer = tf.keras.layers.Normalization()
normalizer.adapt(train_x)

network = models.Sequential()
initializer = tf.keras.initializers.HeNormal()
network.add(layers.SimpleRNN(64, kernel_initializer=initializer, input_shape=(window_size,1), return_sequences=True))
network.add(layers.BatchNormalization())
network.add(layers.Activation("relu"))

network.add(layers.SimpleRNN(32, kernel_initializer=initializer,return_sequences=True))
network.add(layers.BatchNormalization())
network.add(layers.Activation("relu"))

network.add(layers.SimpleRNN(16, kernel_initializer=initializer))
network.add(layers.BatchNormalization())
network.add(layers.Activation("relu"))

network.add(layers.Dense(8, kernel_initializer=initializer))
network.add(layers.BatchNormalization())
network.add(layers.Activation("relu"))

network.add(layers.Dense(3, activation='softmax'))

network.summary()

ACCURACY_THRESHOLD = 0.80
class myCallback(tf.keras.callbacks.Callback): 
    def on_epoch_end(self, epoch, logs={}): 
        if(logs.get('val_accuracy') > ACCURACY_THRESHOLD):   
          print("\nReached %2.2f%% accuracy, so stopping training!!" %(ACCURACY_THRESHOLD*100))   
          self.model.stop_training = True

network.compile(optimizer='Adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
my_callbacks = [
    tf.keras.callbacks.ModelCheckpoint(filepath='model.h5',verbose=1),
    myCallback()
]

history = network.fit(normalizer(train_x), train_labels, validation_split=0.3, callbacks=my_callbacks, epochs=200, batch_size=32)

test_loss, test_acc = network.evaluate(normalizer(test_x), test_labels)

import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training' , 'validation'])
plt.title('Accuracy')
plt.xlabel('epoch')

import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training' , 'validation'])
plt.title('Loss')
plt.xlabel('epoch')

import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import roc_curve, auc

# Plot linewidth.
lw = 2
n_classes = 3
# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

model  = tf.keras.models.load_model('covid_model76.h5')
y_pred_keras = model.predict(normalizer(test_x))

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(test_labels[:, i], y_pred_keras[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(test_labels.ravel(), y_pred_keras.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure(1)
"""
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)
"""
colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multi-class Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


# Zoom in view of the upper left corner.
plt.figure(2)
plt.xlim(0, 0.2)
plt.ylim(0.8, 1)
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Zoom in view of the upper left corner')
plt.legend(loc="lower right")
plt.show()

test_labels[:, 0]
