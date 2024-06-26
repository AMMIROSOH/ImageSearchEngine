from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct
from qdrant_client.models import Distance, VectorParams, Filter, FieldCondition, MatchValue

COLLECTION_NAME = "imagefeatures"

class QdrantStorage:
    def __init__(self, host, port, collectionName):
        """
        Initialize the Qdrant client and connect to the specified collection.
        """
        self.client = QdrantClient(host=host, port=port)
        self.collectionName = collectionName

        # Ensure the collection exists
        self.client.recreate_collection(
            collection_name=self.collectionName,
            vectors_config=VectorParams(size=512, distance=Distance.COSINE),
        )

    def upsertVector(self, productId, vector, metadata):
        """
        Upsert a vector with the given product ID and metadata to the Qdrant collection.
        """
        point = PointStruct(
            id=productId,
            vector=vector,
            payload=metadata
        )
        self.client.upsert(collection_name=self.collectionName, points=[point])

    def deleteCollection(self):
        """
        Delete the Qdrant collection.
        """
        self.client.delete_collection(collection_name=self.collectionName)

    def fetchVector(self, productId):
        """
        Fetch a vector by product ID from the Qdrant collection.
        """
        return self.client.scroll(
            collection_name=self.collectionName,
            scroll_filter={"must": [{"key": "id", "match": {"value": productId}}]}
        )
    def customQuery(self, searchStringFeatures, filters):
        queryFilters = [FieldCondition(key=item["key"], match=MatchValue(value=item["match"]["value"])) for item in filters if item["key"]!=""]
        print(queryFilters)
        queryResponse = []
        if(len(queryFilters)>0):
            queryResponse = self.client.search(
                collection_name=COLLECTION_NAME,
                query_vector=searchStringFeatures,
                query_filter=Filter(
                    must=queryFilters
                ),
                with_payload=True,
                limit=5,
            )
        else:
            queryResponse = self.client.search(
                collection_name=COLLECTION_NAME,
                query_vector=searchStringFeatures,
                with_payload=True,
                limit=5,
            )
        print(queryResponse)
        return [{"score": item.score, "id": item.id, "payload":item.payload} for item in queryResponse]
