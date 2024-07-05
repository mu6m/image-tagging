import os
from flask import Flask, jsonify, request
import uuid
from predict import predict
app = Flask(__name__)


@app.route('/', methods=['POST'])
def generate_tags():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'no files'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        if file:
            path = os.path.join('./uploads/', uuid.uuid4().hex+"-"+file.filename)
            file.save(path)
            pred = predict(path)
            return jsonify(list(pred))

if __name__ == '__main__':
    app.run()

    