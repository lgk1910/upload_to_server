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

print("model.py ver 10")
class Model:
	def __init__(self):
		with open('km_model_10k_classes_20k_imgs.pickle', 'rb') as f:
			self.km = pickle.load(f)

		self.effnet_feature_vec = tf.keras.Sequential([
				hub.KerasLayer("https://hub.tensorflow.google.cn/tensorflow/efficientnet/b0/feature-vector/1",
											trainable=False)
		])
		self.effnet_feature_vec.build([None, 224, 224, 3])  # Batch input shape.
		# self.model = tf.keras.models.load_model('infused_model_4k7.h5', custom_objects={'KerasLayer': hub.KerasLayer})
		self.model = tf.keras.models.load_model('infused_model_4k7.h5')


		self.threshold = 0.99
		
	def get_duplicate_listings(self, URLs, db, Table):
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
		start_time = time.monotonic()
		imgs = np.array(imgs)
		features = self.effnet_feature_vec(imgs).numpy()
		predictions = self.km.predict(features)
		print(predictions)
		input_df = pd.DataFrame(list(zip(input_filenames, predictions)), columns=['input_filename', 'cluster_id'])
		# print(input_df)

		# files = kmean_results_df[kmean_results_df['cluster_id'].isin(predictions)]
		query_return = db.session.query(Table).filter(Table.cluster_id.in_(predictions.tolist()))
		try:
			db.create_all()
		except:
			pass
    
		files = pd.read_sql(query_return.statement, query_return.session.bind)
		print(files)
		merge_df = pd.merge(input_df, files, how='left')
		print(merge_df.dropna())
		if merge_df.dropna().empty:
			output_dict = {}
			return json.dumps(output_dict, indent=3)
		else:
			merge_df = merge_df.dropna()
		print(merge_df)
		img = []
		pairs = []
		for index in range(len(merge_df)):
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
		pairs = np.array(pairs)
		certainties = self.model(pairs)
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


		output_dict = {}
		# dup_found = False
		for id in range(len(avg_certainties['listing'])):
			listing = avg_certainties.iloc[id]['listing']
			pairs = []
			if avg_certainties.iloc[id]['certainty'] >= self.threshold:
				for index in merge_df.index[merge_df['listing']==listing]:
					pairs.append([merge_df.loc[index]['input_filename'], merge_df.loc[index]['file_path'], float(merge_df.loc[index]['certainty'])])
				output_dict[listing] = [len(pairs), float(avg_certainties.iloc[id]['certainty']), tuple(pairs)]
		# print(json.dumps(output_dict, indent=3))
		print('Runtime (seconds): ', time.monotonic() - start_time)
		return json.dumps(output_dict, indent=3)

	def check_duplicate(self, URLs, db, Table):
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
		start_time = time.monotonic()
		imgs = np.array(imgs)
		features = self.effnet_feature_vec(imgs).numpy()
		predictions = self.km.predict(features)
		# print(predictions)

		input_df = pd.DataFrame(list(zip(input_filenames, predictions)), columns=['input_filename', 'cluster_id'])
		# print(input_df)

		# files = kmean_results_df[kmean_results_df['cluster_id'].isin(predictions)]
		query_return = db.session.query(Table).filter(Table.cluster_id.in_(predictions.tolist()))
		try:
			db.create_all()
		except:
			pass
		files = pd.read_sql(query_return.statement, query_return.session.bind)
		# print(files)
		merge_df = pd.merge(input_df, files, how='left')
		# print(merge_df.dropna())
		if merge_df.dropna().empty:
			elapsed_time = time.monotonic() - start_time
			return (False, elapsed_time)
		else:
			merge_df = merge_df.dropna()
			
		img = []
		pairs = []
		for index in range(len(merge_df)):
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
		pairs = np.array(pairs)
		certainties = self.model(pairs)
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
			if avg_certainties.iloc[id]['certainty'] >= self.threshold:
				dup_found = True
				print("Duplicate with listing " + avg_certainties.iloc[id]['listing'])
				break
				# for index in merge_df.index[merge_df['listing']==listing]:
				#   pairs.append([merge_df.loc[index]['input_filename'], merge_df.loc[index]['file_path'], float(merge_df.loc[index]['certainty'])])
				# output_dict[listing] = [len(pairs), float(avg_certainties.iloc[id]['certainty']), tuple(pairs)]
		# print(json.dumps(output_dict, indent=3))
		# print('Runtime (seconds): ', time.monotonic() - start_time)
		elapsed_time = time.monotonic() - start_time
		return (dup_found, elapsed_time)

	def add_to_db(self, listing, URLs, db, Table):
		try:
			os.mkdir(f'dataset/{listing}')
		except:
			pass
 
		imgs = []
		input_filenames = []
		for i, URL in enumerate(URLs):
			try:
				response = requests.get(URL)
				with open(f'dataset/{listing}/img{i}.jpg', "wb") as f:
						f.write(response.content)
				img = cv2.imread(f'dataset/{listing}/img{i}.jpg')
				img = cv2.resize(img, (224, 224,))/255.0
				imgs.append(img)
				input_filenames.append(f'dataset/{listing}/img{i}.jpg')
			except:
				pass
	
		imgs = np.array(imgs)
		features = self.effnet_feature_vec(imgs).numpy()
		predictions = self.km.predict(features)
		for i, input_filename in enumerate(input_filenames):
			row = Table(file_path=input_filename, listing=listing, cluster_id=int(predictions[i]))
			db.session.add(row)
		db.session.commit()
		print("db committed!")