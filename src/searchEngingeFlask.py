from flask import Flask, request, jsonify
from imageEncoder import ClipAI
from qdrantStorage import QdrantStorage, COLLECTION_NAME
from flask_cors import CORS


app = Flask(__name__)
CORS(app)

# Replace with your actual Qdrant database URL
databaseUrl = "http://localhost:6333"

encoder = ClipAI()
qdrantStorage = QdrantStorage('localhost', 6333, COLLECTION_NAME)

@app.route('/search', methods=['POST'])
def search():
    # Parse the JSON request data
    data = request.get_json()
    searchString = data.get('searchString')
    stringFeatures = encoder.encodeText(searchString).cpu().numpy().tolist()[0]

    keyFilters = data.get('keyFilters')

    return jsonify(qdrantStorage.customQuery(stringFeatures, keyFilters))

if __name__ == '__main__':
    app.run(debug=True)
