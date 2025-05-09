# -*- coding: utf-8 -*-
"""ModelCondensed
"""
import numpy as np
import pandas as pd
import glob
import os
from astropy.io import fits
import gc
from scipy.signal import savgol_filter
from sklearn.preprocessing import RobustScaler
import math
from tsfresh import extract_features
from tsfresh.feature_extraction import EfficientFCParameters
from tsfresh.feature_extraction import MinimalFCParameters

#Path to dataset file
directory_path = r"C:\KOI_Q16_long"

#Get the number of files in the directory
num_files = len(os.listdir(directory_path))
print(f"Number of files: {num_files}")

filenames = glob.glob(os.path.join(directory_path, '*.fits'))

def read_files(filenames, batch_size=1000):
    data_list = []
    for i in range(0, len(filenames), batch_size):
        batch_files = filenames[i:i + batch_size]
        for f in batch_files:
            try:
                with fits.open(f, memmap=True) as hdul:
                    data = hdul[1].data
                    data_list.append(data)
            except OSError as e:
                if "Empty or corrupt FITS file" in str(e):
                    print(f"Skipping corrupted file: {f}")
                else:
                    raise e
        gc.collect()
    return data_list

data_list = read_files(filenames)

data_dict = {}
for i, data in enumerate(data_list):
  data_dict[filenames[i]] = pd.DataFrame(data)

#Isolate flux column
columns_to_drop = ["TIME", "PDCSAP_FLUX_ERR", "SAP_QUALITY", "PSF_CENTR1", "PSF_CENTR1_ERR", "PSF_CENTR2", "PSF_CENTR2_ERR", "CADENCENO", "TIMECORR","SAP_FLUX", "SAP_FLUX_ERR", "SAP_BKG","SAP_BKG_ERR","MOM_CENTR1", "MOM_CENTR1_ERR", "MOM_CENTR2", "MOM_CENTR2_ERR", "POS_CORR1", "POS_CORR2"]

for key, df in data_dict.items():
    df.drop(columns=columns_to_drop, inplace=True)
    df.interpolate(inplace=True)
    df.drop(0, axis=0, inplace=True)
    df.ffill()
    df.bfill()
    df.dropna(axis = 0, inplace = True)
    flux_data = df['PDCSAP_FLUX']
    smoothed_flux = savgol_filter(flux_data, window_length=11, polyorder=2)
    cleaned_flux = flux_data - smoothed_flux
    df['PDCSAP_FLUX'] = cleaned_flux

    #Downsample
    n_pts = 2000
    x_old = np.arange(len(df))
    x_new = np.linspace(0, len(df) - 1, n_pts)
    flux_down = np.interp(x_new, x_old, df['PDCSAP_FLUX'].values)
    data_dict[key] = pd.DataFrame({'PDCSAP_FLUX': flux_down})

num_items = math.ceil(len(data_dict) / 4)
test_dict = dict(list(data_dict.items())[:num_items])
train_dict = dict(list(data_dict.items())[num_items:])
#Feature extraction
def prepinput(data_dict):
    all_samples = []
    for idx, (key, df) in enumerate(data_dict.items()):
        temp_df = df.copy()
        temp_df['id'] = idx
        all_samples.append(temp_df)
    combined_df = pd.concat(all_samples, ignore_index=True)
    return combined_df
#Drop low-impact features
extraction_settings = EfficientFCParameters()
for feat in [
    "fft_coefficient",
    "large_standard_deviation",
    "percentage_of_reoccurring_values_to_all_values"
]: 
    if feat in extraction_settings:
        extraction_settings.pop(feat)

min_params = MinimalFCParameters()
for feat, func in min_params.items():
    extraction_settings[feat] = func
train_combined = prepinput(train_dict)
test_combined = prepinput(test_dict)
train_features = extract_features(train_combined, column_id='id', column_value='PDCSAP_FLUX', default_fc_parameters=extraction_settings, impute_function=None, n_jobs=1)
test_features = extract_features(test_combined, column_id='id', column_value='PDCSAP_FLUX', default_fc_parameters=extraction_settings, impute_function=None, n_jobs=1)
train_features = train_features.fillna(0)
test_features = test_features.fillna(0)
scaler = RobustScaler()
train_scaled = scaler.fit_transform(train_features)
test_scaled = scaler.transform(test_features)

#Make an artificial no-transit dataset
num_samples = 1000  # Number of flat lightcurves to generate
length = 2000      # Length of each lightcurve

artificial_data = np.ones((num_samples, length))
noise = np.random.normal(0, 0.01, artificial_data.shape)
artificial_data = artificial_data =+ noise
artificial_samples = []
for idx in range(num_samples):
    df_temp = pd.DataFrame({'PDCSAP_FLUX': artificial_data[idx]})
    df_temp['id'] = idx  # Assign unique id for TSFresh grouping
    artificial_samples.append(df_temp)
artificial_combined = pd.concat(artificial_samples, ignore_index=True)
artificial_features = extract_features(artificial_combined, column_id='id', column_value='PDCSAP_FLUX', default_fc_parameters=extraction_settings, impute_function=None,n_jobs=1)
artificial_features = artificial_features.fillna(0)
artificial_scaled = scaler.transform(artificial_features)

#Combine datasets
combined_data = np.concatenate((artificial_scaled, test_scaled), axis=0)
#Get labels
no_transit_labels = np.full(artificial_data.shape[0], -1)
transit_labels = np.full(test_scaled.shape[0], 1)
#Combine labels
combined_labels = np.concatenate((no_transit_labels, transit_labels), axis=0)

#Train SVM
from datetime import datetime, timezone
from sklearn.svm import OneClassSVM
t0 = datetime.now(timezone.utc)

model = OneClassSVM(kernel="rbf", gamma='auto', nu=0.05)
model.fit(train_scaled)

t1 = datetime.now(timezone.utc)
time_delta = t1 - t0
print(f"Time taken: {time_delta.total_seconds()} seconds")

#Run predictions
from datetime import datetime, timezone
t0 = datetime.now(timezone.utc)

predictions = model.predict(combined_data)
t1 = datetime.now(timezone.utc)
time_delta = t1 - t0
print(f"Time taken: {time_delta.total_seconds()} seconds")

class_labels = {-1: 'no_planet', 1: 'planet'}
results = np.vectorize(class_labels.get)(predictions)

count1 = sum(x == "no_planet" for x in results)
print('Number of Non-planets:', count1)
count2 = sum(x == "planet" for x in results)
print('Number of Planets:', count2)

#Calculate Accuracy, F1

correct_predictions = (predictions == combined_labels)
accuracy = np.sum(correct_predictions) / len(correct_predictions)
print("Accuracy:", accuracy)
from sklearn.metrics import f1_score
f1 = f1_score(combined_labels, predictions, pos_label= 1)
print(f1)
