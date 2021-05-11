from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
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

