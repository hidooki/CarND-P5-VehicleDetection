"""Train an SVM classifier and pickle the resulting model parameters"""
from functools import partial
import glob
import numpy as np
import pickle
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from features import feature_extractor
from utils import read_image


# Parameters to try
resize = (64, 64)
color_space = 'YCrCb'
spatial_size = (32, 32)
histbin = 32
orient = 8
pix_per_cell = 8
cell_per_block = 2
channels = [0, 1, 2]

# Feature extrating function
extract_features = partial(
    feature_extractor,
    resize=resize,
    color_space=color_space,
    spatial_size=spatial_size,
    hist_bins=histbin,
    hist_range=None,
    hog_orient=orient,
    hog_pix_per_cell=pix_per_cell,
    hog_cell_per_block=cell_per_block,
    hog_channels=channels,
    spatial_features=True,
    color_features=True,
    hog_features=True)


# Read in car and non-car images
cars = glob.glob('images/vehicles/*/*.png')
notcars = glob.glob('images/non-vehicles/*/*.png')

print('Extracting features from ', len(cars), ' cars and ',
      len(notcars), ' notcars')

tic = time.time()

car_features = np.vstack([extract_features(read_image(c)) for c in cars])
notcar_features = np.vstack([extract_features(read_image(c)) for c in notcars])

toc = time.time()
print(round(toc - tic, 2), 'Seconds to extract features...')


# Create an array stack of feature vectors
X = np.vstack((car_features, notcar_features)).astype(np.float64)
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

# Split up data into randomized training and test sets
# rand_state = np.random.randint(0, 100)
rand_state = 90
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)

print('Feature vector length:', len(X_train[0]))
# Use a linear SVC
svc = LinearSVC()
# Check the training time for the SVC
tic = time.time()
svc.fit(X_train, y_train)
toc = time.time()
print(round(toc - tic, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Training Accuracy of SVC = ', round(svc.score(X_train, y_train), 4))
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

# Check the prediction time for a single sample
tic = time.time()
n_predict = 10
print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
print('For these', n_predict, 'labels: ', y_test[0:n_predict])
toc = time.time()
print(round(toc - tic, 5), 'Seconds to predict', n_predict, 'labels with SVC')


params = {
    'resize': resize,
    'color_space': color_space,
    'spatial_size': spatial_size,
    'histbin': histbin,
    'orient': orient,
    'pix_per_cell': pix_per_cell,
    'cell_per_block': cell_per_block,
    'channels': channels}

# pickle.dump(extract_features2, open('params/extractor.p', 'wb'))
pickle.dump(params, open('params/params.p', 'wb'))
pickle.dump(svc, open('params/classifier.p', 'wb'))
pickle.dump(X_scaler, open('params/feature_scaler.p', 'wb'))
