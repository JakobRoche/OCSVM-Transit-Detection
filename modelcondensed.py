# -*- coding: utf-8 -*-
"""ModelCondensed
"""
!pip install scipy
import numpy as np
!pip install astropy
from astropy.table import Table
import pandas as pd
import glob
import os
!pip install astropy.io.fits
import astropy.io.fits as fits
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler

# Path to dataset file
directory_path = "/content/KOI_Q16_long"

# Get the number of files in the directory
num_files = len(os.listdir(directory_path))
print(f"Number of files: {num_files}")

# Get filenames
filenames = glob.glob(os.path.join(directory_path, '*.fits'))

# Read the files
def read_files(path=directory_path):
  data_list = []
  for file in os.listdir(path):
      try:
          hdulist = fits.open(os.path.join(path, file), ignore_missing_simple=True)
          data = hdulist[1].data
          data_list.append(data)
      except OSError as e:
          if "Empty or corrupt FITS file" in str(e):
              print(f"Skipping corrupted file: {file}")
          else:
              raise e
  return data_list
data_list = read_files(directory_path)

# Create dataset dictionary
data_dict = {}
for i, data in enumerate(data_list):
  data_dict[filenames[i]] = pd.DataFrame(data)

# Preprocess Files
# Isolate the flux column
columns_to_drop = ["TIME", "PDCSAP_FLUX_ERR", "SAP_QUALITY", "PSF_CENTR1", "PSF_CENTR1_ERR", "PSF_CENTR2", "PSF_CENTR2_ERR", "CADENCENO", "TIMECORR","SAP_FLUX", "SAP_FLUX_ERR", "SAP_BKG","SAP_BKG_ERR","MOM_CENTR1", "MOM_CENTR1_ERR", "MOM_CENTR2", "MOM_CENTR2_ERR", "POS_CORR1", "POS_CORR2"]

# Iterate over the dictionary and process the files
for df in data_dict.values():
    df.drop(columns=columns_to_drop, inplace=True)
    df.interpolate(inplace=True)
    df.drop(0, axis=0, inplace=True)

    flux_data = df['PDCSAP_FLUX']
    smoothed_flux = savgol_filter(flux_data, window_length=50, polyorder=2)
    cleaned_flux = flux_data - smoothed_flux

    df['PDCSAP_FLUX'] = cleaned_flux

# Print the modified files (Optional)
#for filename, df in data_dict.items():
#    print(f"Dataframe for file {filename}:")
#    print(df)
#    print(df.isnull().sum())

# Do a train-test split
import math
num_items = math.ceil(len(data_dict) / 4)

# Create a new dictionary with the split-off items
test_dict = {}
for i, (key, value) in enumerate(data_dict.items()):
    if i < num_items:
        test_dict[key] = value
# Print the number of items in the test set
len(test_dict)

# Create a new dictionary without the transferred files
new_data_dict = {key: value for key, value in data_dict.items() if key not in test_dict}

# Assign the new dictionary to data_dict
data_dict = new_data_dict
# Print number of test set files
len(data_dict)

# Process datasets into a usable form

# Get filenames for the datasets
filenames2 = list(data_dict.keys())
filenames3 = list(test_dict.keys())
# Create new dictionaries
train_set = []
for i in range(len(data_dict)):
  train_set.append(data_dict[filenames2[i]])
test_set = []
for i in range(len(test_dict)):
  test_set.append(test_dict[filenames3[i]])

train_set2 = []
for df in train_set:
    train_set2.append(df.values)
# Turn the training set into an array
train_set2 = np.array(train_set2)
# Reshape all_data to have 2 dimensions
train_set_2d = train_set2.reshape(train_set2.shape[0], -1)
# Scale the data
scaler = StandardScaler()
train_set_2d = scaler.fit_transform(train_set_2d)
# Do the same for the test set

test_set2 = []
for df in test_set:
    test_set2.append(df.values)

test_set2 = np.array(test_set2)
test_set_2d = test_set2.reshape(test_set2.shape[0], -1)
test_set_2d = scaler.fit_transform(test_set_2d)

# Train the SVM

from datetime import datetime
from sklearn.svm import OneClassSVM

# Get the current time before the code block.
t0 = datetime.utcnow()
model = OneClassSVM(kernel="rbf", gamma='scale', nu=0.05)
model.fit(train_set_2d)

# Get the current time after the code block.
t1 = datetime.utcnow()

# Calculate the difference between the two times.
time_delta = t1 - t0

# Print the time difference in seconds.
print(f"Time taken: {time_delta.total_seconds()} seconds")

# Make an artificial no-transit dataset
num_samples = 1000  # Number of flat lightcurves to generate
length = 4202       # Length of each lightcurve

# Generate the artificial dataset
artificial_data = np.ones((num_samples, length))

# Add some noise to make it more realistic
noise_level = 0.01
noise = np.random.normal(0, noise_level, artificial_data.shape)
artificial_data += noise

# Apply preprocessing to artificial dataset
scaler = StandardScaler()
smoothed_aflux = savgol_filter(artificial_data, window_length=50, polyorder=2)
cleaned_aflux = artificial_data - smoothed_aflux
scaled_aflux = scaler.fit_transform(cleaned_aflux)
artificial_data_reshaped = scaled_aflux.reshape(num_samples, -1)

# Combine the datasets
combined_data = np.concatenate((artificial_data_reshaped, test_set_2d), axis=0)
# Get labels
no_transit_labels = np.full(artificial_data_reshaped.shape[0], -1)
transit_labels = np.full(test_set_2d.shape[0], 1)
# Combine the labels
combined_labels = np.concatenate((no_transit_labels, transit_labels), axis=0)

# Predict on test set

from datetime import datetime

# Get the current time before the code block.
t0 = datetime.utcnow()

predictions = model.predict(combined_data)

# Get the current time after the code block.
t1 = datetime.utcnow()

# Calculate the difference between the two times.
time_delta = t1 - t0

# Print the time difference in seconds.
print(f"Time taken: {time_delta.total_seconds()} seconds")

# Turn the predictions into natural language
class_labels = {-1: 'no_planet', 1: 'planet'}
results = np.vectorize(class_labels.get)(predictions)
print(results)

count1 = sum(x == "no_planet" for x in results)
print('Number of Non-planets:', count1)
count2 = sum(x == "planet" for x in results)
print('Number of Planets:', count2)

# Calculate model accuracy
correct_predictions = (predictions == combined_labels)
accuracy = np.sum(correct_predictions) / len(correct_predictions)
print("Accuracy:", accuracy)

# Calculate model F1 score
from sklearn.metrics import f1_score

f1 = f1_score(combined_labels, predictions, pos_label= 1)

# Print the F1 score
print(f1)