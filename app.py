from flask import Flask, request, jsonify

app = Flask(__name__)

def index():
    return jsonify({"status":"success"}), 200