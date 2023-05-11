from flask import Flask, request
from pymongo import MongoClient
from gridfs import GridFS
from io import BytesIO
from keras.utils import img_to_array
from keras.models import load_model
from PIL import Image
import numpy as np

app = Flask(__name__)

# Load the saved model
model = load_model("my_model.h5")

# Define the classes
class_names = ['healthy', 'unhealthy']

# Establish a connection to MongoDB
client = MongoClient("mongodb+srv://mateusok:test123@cluster0.9yg69ez.mongodb.net/?retryWrites=true&w=majority")
db = client['eyeClassifier']

# Create a GridFS object
fs = GridFS(db, collection='images')


@app.route('/upload', methods=['POST'])
def image_upload():
    # Read the image file
    file = request.files['file']
    image = Image.open(BytesIO(file.read()))
    image = image.resize((224, 224))
    image_array = img_to_array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Make a prediction
    prediction = model.predict(image_array)
    predicted_class_index = np.argmax(prediction)
    predicted_class = class_names[predicted_class_index]
    confidence = prediction[0][predicted_class_index] * 100

    # Save the image to MongoDB GridFS
    image_data = BytesIO()
    image.save(image_data, format='JPEG')
    image_data.seek(0)
    file_id = fs.put(image_data, filename=file.filename)

    # Create a new document with the analysis results
    analysis_result = {
        'file_id': str(file_id),
        'predicted_class': predicted_class,
        'confidence': confidence
    }

    # Save the analysis result to MongoDB
    analysis_collection = db['resultados']
    analysis_collection.insert_one(analysis_result)

    return {'file_id': str(file_id), 'predicted_class': predicted_class, 'confidence': confidence}


@app.route('/result/<file_id>', methods=['GET'])
def image_result(file_id):
    # Retrieve the analysis result from MongoDB
    analysis_collection = db['resultados']
    result = analysis_collection.find_one({'file_id': file_id})
    if result:
        return {'predicted_class': result['predicted_class'], 'confidence': result['confidence']}
    else:
        return {'message': 'Analysis result not found.'}


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
