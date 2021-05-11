from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
import re
import os
from model import Model
import time
import requests
import numpy as np

try:
	os.mkdir('dataset')
	print("Directory dataset created")
except:
	pass

try:
	os.mkdir('test_images')
	print("Directory test_images created")
except:
	pass
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///main.db'
db = SQLAlchemy(app)

class Table(db.Model):
	id = db.Column(db.Integer, primary_key=True)
	file_path=db.Column(db.String(200), unique=True, nullable=False)
	listing=db.Column(db.String(200), nullable=False)
	cluster_id=db.Column(db.Integer, nullable=False)

	def __repr__(self):
		return f"Object ('{self.file_path}', '{self.listing}', '{self.cluster_id}')"

	
@app.route('/')
def home():
	return "Real estate image Tagging using Transfer Learning model"

@app.route("/add/", methods=['GET', 'POST'])
def add():
	def check_url(url):
		# Compile the ReGex
		p = re.compile(r'^(?:http|ftp)s?://' # http:// or https://
								 r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|' # domain...
								 r'localhost|' # localhost...
								 r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}|' # ...or ipv4
								 r'\[?[A-F0-9]*:[A-F0-9:]+\]?)' # ...or ipv6
								 r'(?::\d+)?' # optional port
								 r'(?:/?|[/?]\S+)$', re.IGNORECASE)

		# If the string is empty 
		# return false
		if (url == None):
			return False
	
		# Return if the string 
		# matched the ReGex
		if(re.search(p, url)):
			return True
		else:
			return False
	
	# execution_time = time.time()
	try:
		content = request.json
		url_list = content["urls"]
		listing = content["listing"]
		url_list_filtered = []
		for url in url_list:
			if check_url(str(url))==True:
				url_list_filtered.append(url)
	
		print(url_list_filtered)
		dup_found, elapsed_time = trained_model.check_duplicate(url_list_filtered, db, Table)
		start_time = time.monotonic()
		if dup_found == False:
			trained_model.add_to_db(listing, url_list, db, Table)
			# # Add new listing to database
			# try:
			# 	os.mkdir(f'dataset/{listing}')
			# 	print(f'dataset/{listing} made')
			# except:
			# 	pass
	
			# imgs = []
			# input_filenames = []
			# for i, URL in enumerate(url_list):
			# 	try:
			# 		response = requests.get(URL)
			# 		with open(f'dataset/{listing}/img{i}.jpg', "wb") as f:
			# 				f.write(response.content)
			# 		img = cv2.imread(f'dataset/{listing}/img{i}.jpg')
			# 		img = cv2.resize(img, (224, 224,))/255.0
			# 		imgs.append(img)
			# 		input_filenames.append(f'dataset/{listing}/img{i}.jpg')
			# 	except:
			# 		pass
			# imgs = np.array(imgs)
			# features = effnet_feature_vec(imgs).numpy()
			# predictions = km.predict(features)
			# for i, input_filename in enumerate(input_filenames):
			# 	row = Table(file_path=input_filename, listing=listing, cluster_id=int(predictions[i]))
			# 	db.session.add(row)
			# db.session.commit()
			# print("db committed!!!")
		else:
			pass
		elapsed_time += time.monotonic() - start_time

		return jsonify(listing=content['listing'], duplicate=dup_found, time=elapsed_time), 200
	except Error as e:
		# print(e)
		return jsonify(label="Error"), 400

@app.route("/duplicate_check/", methods=['GET', 'POST'])
def check():
	print("here")
	def check_url(url):
		# Compile the ReGex
		p = re.compile(r'^(?:http|ftp)s?://' # http:// or https://
		r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|' # domain...
		r'localhost|' # localhost...
		r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}|' # ...or ipv4
		r'\[?[A-F0-9]*:[A-F0-9:]+\]?)' # ...or ipv6
		r'(?::\d+)?' # optional port
		r'(?:/?|[/?]\S+)$', re.IGNORECASE)
	
		# If the string is empty 
		# return false
		if (url == None):
			return False
	
		# Return if the string 
		# matched the ReGex
		if(re.search(p, url)):
			return True
		else:
			return False
	try:
		print("tryyyyyyyyyyyyy")
		content = request.json
		url_list = content["urls"]
		url_list_filtered = []
		for url in url_list:
			if check_url(str(url))==True:
				url_list_filtered.append(url)
	
		print(url_list_filtered)
		result = trained_model.get_duplicate_listings(url_list_filtered, db, Table)
		return result
	except:
		return jsonify(label="Error"), 400

if __name__ == '__main__':
	trained_model = Model()
	app.run()
