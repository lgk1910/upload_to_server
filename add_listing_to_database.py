import pandas as pd
from matplotlib import pyplot as plt
import pickle
import numpy as np
import glob
import requests
import cv2
import tensorflow as tf
import tensorflow_hub as hub
import keras
import skimage
import random
import string
import PIL
import re
import json
import time
import os

from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from database import Table, db, app

try:
    os.mkdir('dataset')
except:
    pass
try:
    os.mkdir('test_images')
except:
    pass
db.create_all()

with open('km_model_10k_classes_20k_imgs.pickle', 'rb') as f:
  km = pickle.load(f)

effnet_feature_vec = tf.keras.Sequential([
    hub.KerasLayer("https://hub.tensorflow.google.cn/tensorflow/efficientnet/b0/feature-vector/1",
                   trainable=False)
])
effnet_feature_vec.build([None, 224, 224, 3])  # Batch input shape.
model = tf.keras.models.load_model('infused_model_4k7.h5', custom_objects={'KerasLayer': hub.KerasLayer})

threshold = 0.99
def check_duplicate(URLs):
  start_time = time.monotonic()
  imgs = []
  input_filenames = []
  for i, URL in enumerate(URLs):
    try:
      response = requests.get(URL)
      with open(f'test_images/img{i}.jpg', "wb") as f:
          f.write(response.content)
      img = cv2.imread(f'test_images/img{i}.jpg')
      img = cv2.resize(img, (224, 224,))/255.0
      imgs.append(img)
      input_filenames.append(f'test_images/img{i}.jpg')
    except:
      pass
  imgs = np.array(imgs)
  features = effnet_feature_vec(imgs).numpy()
  predictions = km.predict(features)
  # print(predictions)

  input_df = pd.DataFrame(list(zip(input_filenames, predictions)), columns=['input_filename', 'cluster_id'])
  # print(input_df)

  # files = kmean_results_df[kmean_results_df['cluster_id'].isin(predictions)]
  query_return = db.session.query(Table).filter(Table.cluster_id.in_(predictions.tolist()))
  files = pd.read_sql(query_return.statement, query_return.session.bind)
  # print(files)
  merge_df = pd.merge(input_df, files, how='left')
  # print(merge_df.dropna())
  if merge_df.dropna().empty:
    return False
  else:
    merge_df = merge_df.dropna()
    
  img = []
  pairs = []
  for index in range(len(merge_df)):
    try:
      if index > 0:
        if merge_df.iloc[index]['input_filename'] != merge_df.iloc[index-1]['input_filename']:
          img = cv2.imread(merge_df.iloc[index]['input_filename'])
          img = cv2.resize(img, (224, 224))/255.0
      else:
        img = cv2.imread(merge_df.iloc[index]['input_filename'])
        img = cv2.resize(img, (224, 224))/255.0
      matching_img = cv2.imread(merge_df.iloc[index]['file_path'])
      matching_img = cv2.resize(matching_img, (224, 224))/255.0
      pair = np.vstack((img, matching_img))
      pairs.append(pair.tolist())
    except:
      # merge_df = merge_df.drop(labels=index, axis=0)
      pass
  pairs = np.array(pairs)
  certainties = model(pairs)
  merge_df['certainty'] = certainties[:, 1]
  # merge_df = merge_df[merge_df['certainty']>=threshold]

  # Filter out some listing that has only 1 duplicate image
  listing_count = merge_df.groupby('listing', as_index=False).count()
  listing_count = listing_count[listing_count['certainty'] >= 2]['listing'].tolist()
  # print(listing_count)
  merge_df = merge_df[merge_df['listing'].isin(listing_count)]
  # print(merge_df)

  avg_certainties = merge_df.groupby('listing', as_index=False).mean()[['listing','certainty']].sort_values(by=['certainty'],ascending=False)
  # print(avg_certainties)


  # output_dict = {}
  dup_found = False
  for id in range(len(avg_certainties['listing'])):
    listing = avg_certainties.iloc[id]['listing']
    pairs = []
    if avg_certainties.iloc[id]['certainty'] >= threshold:
      dup_found = True
      print("Duplicate with listing " + avg_certainties.iloc[id]['listing'])
      break
      # for index in merge_df.index[merge_df['listing']==listing]:
      #   pairs.append([merge_df.loc[index]['input_filename'], merge_df.loc[index]['file_path'], float(merge_df.loc[index]['certainty'])])
      # output_dict[listing] = [len(pairs), float(avg_certainties.iloc[id]['certainty']), tuple(pairs)]
  # print(json.dumps(output_dict, indent=3))
  # print('Runtime (seconds): ', time.monotonic() - start_time)
  return dup_found

listing_df = pd.read_csv('listing_and_images.csv')
columns = listing_df.columns

# Start adding new listing
for id in range(len(listing_df)):
  print("Adding listing " + listing_df.iloc[id][0] + "...")
  # Check if listing existed
  if os.path.isdir(f'dataset/{listing_df.iloc[id][0]}'):
    print(f"Listing {listing_df.iloc[id][0]} has existed in database")
    continue

  input_URLs = []
  for column in columns[1:]:
    if pd.isnull(listing_df.iloc[id][column]):
      break
    input_URLs.append(listing_df.iloc[id][column])
  print(input_URLs)
  # break
  # Check duplicate
  if check_duplicate(input_URLs):
    print("Duplicate listing detected. Adding fail.")
  else:
    print(f"New listing. Adding...")
    try:
      os.mkdir(f'dataset/{listing_df.iloc[id][0]}')
      imgs = []
      input_filenames = []
      for i, URL in enumerate(input_URLs):
        try:
          response = requests.get(URL)
          with open(f'dataset/{listing_df.iloc[id][0]}/img{i}.jpg', "wb") as f:
              f.write(response.content)
          img = cv2.imread(f'dataset/{listing_df.iloc[id][0]}/img{i}.jpg')
          img = cv2.resize(img, (224, 224,))/255.0
          imgs.append(img)
          input_filenames.append(f'dataset/{listing_df.iloc[id][0]}/img{i}.jpg')
        except:
          pass
      imgs = np.array(imgs)
      features = effnet_feature_vec(imgs).numpy()
      predictions = km.predict(features)
      for i, input_filename in enumerate(input_filenames):
        row = Table(file_path=input_filename, listing=listing_df.iloc[id][0], cluster_id=int(predictions[i]))
        db.session.add(row)
      db.session.commit()
    except:
      print(f"Listing {listing_df.iloc[id][0]} has existed in database")
    print("Adding successfully")
  print("---------------------")