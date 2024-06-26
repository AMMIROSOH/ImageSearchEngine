import json
import h5py
import numpy as np
from imageEncoder import ClipAI
from qdrantStorage import QdrantStorage, COLLECTION_NAME

def main():
    with open('Data\\products.json', 'r', encoding='utf-8') as f:
        products = json.load(f)
        
    qdrantStorage = QdrantStorage('localhost', 6333, COLLECTION_NAME)
    encoder = ClipAI()

    # Process & Store encoded features in Pinecone
    ids = []
    vectors = []
    for i, product in enumerate(products):
        productId = product['id']
        if(len(product['images'])>0):
            vector = encoder.processProduct(product)[0]
            # max length of clip ai string text encoder
            text = product['description'] if len(product['description']) < 77 else product['description'][0:77]
            textVector = encoder.encodeText(text).cpu().numpy().tolist()[0]
            metadata = {key: value for key, value in product.items()}
            
            qdrantStorage.upsertVector(productId, vector, textVector, metadata)
            vectors.append(vector)
            ids.append(ids)
            print(i) if i%100==0 else 0

    print("Encoded features stored in Qdrant.")

    with h5py.File(f'Data\\vectors.h5', 'w') as f:
        f.create_dataset(f'vectors', data=np.array(vectors))
    with h5py.File(f'Data\\ids.h5', 'w') as f:
        f.create_dataset(f'ids', data=np.array(ids))


if __name__ == "__main__":
    main()
