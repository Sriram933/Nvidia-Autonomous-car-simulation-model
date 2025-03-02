# Autonomous Vehicle Steering Prediction using Deep Learning

This project implements a deep learning model for autonomous vehicle steering angle prediction. The model is trained on a dataset of driving images and corresponding steering angles to predict optimal steering control for a self-driving car.
## Features
- Loads and preprocesses driving images from a dataset
- Balances the dataset by equalizing steering angle distributions
- Augments images using translation, zoom, brightness adjustments, and flipping
- Uses a Convolutional Neural Network (CNN) to predict steering angles
- Implements real-time inference using a Flask and SocketIO server
- Saves and loads trained models for deployment

## Dataset

The dataset used for training consists of images captured from a driving simulator, stored in a CSV file (driving_log.csv). It contains the following columns:
- Center: Path to the center camera image
- Left: Path to the left camera image
- Right: Path to the right camera image
- Steering: Steering angle values
- Throttle: Throttle values
- Brake: Brake values
- Speed: Vehicle speed

## Data Preprocessing & Augmentation
### Extract filenames from paths:
```python
def getname(X):
    return X.split('\\')[-1]

data['Center'] = data['Center'].apply(getname)
```
### Balance the dataset by limiting the number of samples per steering angle:
```python
from sklearn.utils import shuffle

def balanceData(data, display=True):
    nBins = 31
    samplesPerBin = 1500
    hist, bins = np.histogram(data['Steering'], nBins)
    
    removeindexlist = []
    for j in range(nBins):
        bindatalist = [i for i in range(len(data)) if bins[j] <= data['Steering'][i] <= bins[j+1]]
        bindatalist = shuffle(bindatalist)
        bindatalist = bindatalist[samplesPerBin:]
        removeindexlist.extend(bindatalist)
    
    data.drop(data.index[removeindexlist], inplace=True)
    return data

data = balanceData(data)
```
### Image augmentation to improve model generalization:
```python
from imgaug import augmenters as iaa

def augmentImage(imgPath, steering):
    img = mpimg.imread(imgPath)
    if np.random.rand() < 0.5:
        img = iaa.Affine(translate_percent={'x': (-0.1, 0.1), 'y': (-0.1, 0.1)}).augment_image(img)
    if np.random.rand() < 0.5:
        img = iaa.Affine(scale=(1, 1.2)).augment_image(img)
    if np.random.rand() < 0.5:
        img = iaa.Multiply((0.4, 1.2)).augment_image(img)
    if np.random.rand() < 0.5:
        img = cv2.flip(img, 1)
        steering = -steering
    return img, steering
```
## Model Architecture
### The CNN used is inspired by NVIDIA's End-to-End Self-Driving Car Model:
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

model = Sequential()
model.add(Conv2D(24, (5,5), strides=(2,2), input_shape=(66,200,3), activation='elu'))
model.add(Conv2D(36, (5,5), strides=(2,2), activation='elu'))
model.add(Conv2D(48, (5,5), strides=(2,2), activation='elu'))
model.add(Conv2D(64, (3,3), activation='elu'))
model.add(Conv2D(64, (3,3), activation='elu'))
model.add(Flatten())
model.add(Dense(100, activation='elu'))
model.add(Dense(50, activation='elu'))
model.add(Dense(10, activation='elu'))
model.add(Dense(1))

model.compile(Adam(lr=0.0001), loss='mse')
```
### Training the Model
The model is trained using a batch generator to efficiently load and process images:
```python
history = model.fit(batchgenerator(X_train, y_train, 100, 1),
                    steps_per_epoch=300,
                    epochs=10,
                    validation_data=batchgenerator(X_val, y_val, 100, 0),
                    validation_steps=200)
```
Plot the loss curve:
```python
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('Loss')
plt.xlabel('Epoch')
model.save('model.h5')
```
### Real-Time Steering Control
A Flask & SocketIO server is used to run real-time inference from a simulation environment:
```python
import socketio
import eventlet
from flask import Flask
from PIL import Image
import base64
from io import BytesIO

sio = socketio.Server()
app = Flask(__name__)

@sio.on('telemetry')
def telemetry(sid, data):
    image = Image.open(BytesIO(base64.b64decode(data['image'])))
    image = np.asarray(image)
    image = preProcess(image)
    image = np.array([image])
    steering = float(model.predict(image))
    throttle = 1.0 - float(data['speed']) / 10
    sendControl(steering, throttle)

def sendControl(steering, throttle):
    sio.emit('steer', {'steering_angle': str(steering), 'throttle': str(throttle)})

if __name__ == '__main__':
    model = load_model('model.h5')
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('localhost', 4567)), app)
```
### Train the model: 
```python
python train.py
```
### Start the inference server:
```python
python drive.py
```
Connect the simulator to localhost and observe the self-driving behavior.








