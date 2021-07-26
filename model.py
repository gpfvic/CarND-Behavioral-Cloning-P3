import os, csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout,SpatialDropout2D
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.utils import plot_model
import sklearn
from sklearn.model_selection import train_test_split

DIR = './data/'
lines = []
images = []
angles = []

with open (os.path.join(DIR+'driving_log.csv')) as f:
    reader = csv.reader(f)
    for line in reader:
        lines.append(line)

# to correct the angle
correction_factor = 0.2

# Read images, steerings
for line in lines[1:]:
    for idx in range(3):
        if idx==0:
            angle = float(line[3]) 
        if idx==1:
            angle = float(line[3]) + correction_factor
        if idx==2:
            angle = float(line[3]) - correction_factor
        angles.append(angle)
        fpath = line[idx]
        fname = os.path.basename(fpath)
        img = mpimg.imread(os.path.join(DIR, 'IMG/', fname))
        images.append(img)

# split the samples for training and validation by 80:20
samples = list(zip(images, angles))
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

print('Train samples: {}'.format(len(train_samples)))
print('Validation samples: {}'.format(len(validation_samples)))

# training data generator
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: 
        samples = sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset : offset+batch_size]
            imagess = []
            angles = []
            for img, angle in batch_samples:
                imagess.append(img)
                angles.append(angle)
                # for data augmentation by flip the image and negative the angle
                imagess.append(np.fliplr(img))
                angles.append(-1.0*angle)
                
            X_train = np.array(imagess)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# Model architecture
model = Sequential()
model.add(Cropping2D(cropping=((60,20), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: (x / 255.0) - 0.5))
model.add(Conv2D(filters=24, kernel_size=5, strides=2, activation='relu'))
model.add(SpatialDropout2D(0.2))
model.add(Conv2D(filters=36, kernel_size=5, strides=2, activation='relu'))
model.add(SpatialDropout2D(0.2))
model.add(Conv2D(filters=48, kernel_size=5, strides=2, activation='relu'))
model.add(SpatialDropout2D(0.3))
model.add(Conv2D(filters=64, kernel_size=3, strides=1, activation='relu'))
model.add(SpatialDropout2D(0.3))
model.add(Conv2D(filters=64, kernel_size=3, strides=1, activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.4))
model.add(Dense(50))
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Dense(1))
model.summary()

# plot the model architecture to image
plot_model(model, to_file='model.png',show_shapes=True)

# for tensorboard visualization
checkpoint = ModelCheckpoint(
    filepath="model_epoch{epoch:02d}-loss{val_loss:.4f}.h5",
    monitor='val_loss',
    save_best_only=True,
)

# Training parameters 
BATCH_SIZE = 256
EPOCHS = 10

model.compile(loss='mse', optimizer=Adam())

train_generator = generator(train_samples, batch_size=BATCH_SIZE)
validation_generator = generator(validation_samples, batch_size=BATCH_SIZE)

history_object = model.fit_generator(
    train_generator,
    steps_per_epoch = len(train_samples)/BATCH_SIZE, 
    validation_data = validation_generator,
    validation_steps = len(validation_samples)/BATCH_SIZE, 
    epochs = EPOCHS, 
    callbacks = [checkpoint, TensorBoard(log_dir="./logs")],
    verbose = 1
)

model.save('model.h5')
### print the keys contained in the history object
print(history_object.history.keys())


### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig("./visualize_loss.png")