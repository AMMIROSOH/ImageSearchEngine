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
        
        if not self.collectionExists(self.collectionName):
            self.client.recreate_collection(
                collection_name=self.collectionName,
                vectors_config={
                    "text": VectorParams(
                        size=512,
                        distance=Distance.EUCLID,
                    ),
                    "image": VectorParams(
                        size=512,
                        distance=Distance.COSINE,
                    ),
                },
            )
    def collectionExists(self, collectionName):
        return collectionName in [item[1][0].name for item in self.client.get_collections()]

    def upsertVector(self, productId, imageVector, textVector, metadata):
        """
        Upsert a vector with the given product ID and metadata to the Qdrant collection.
        """
        point = PointStruct(
            id=productId,
            vector={
                "image": imageVector,
                "text": textVector
            },
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
    def rawQuery(self, searchStringFeatures, filters, vectorName="image", limit=5):
        queryResponse = self.client.search(
            collection_name=COLLECTION_NAME,
            query_vector=(vectorName, searchStringFeatures),
            query_filter=Filter(
                must=filters
            ),
            with_payload=True,
            limit=limit,
        )
        return queryResponse
    def customQuery(self, searchStringFeatures, filters):
        imageW = 1
        textW = 0.3
        queryFilters = [FieldCondition(key=item["key"], match=MatchValue(value=item["match"]["value"])) for item in filters if item["key"]!=""]
        print(queryFilters)
        responseImage = self.rawQuery(searchStringFeatures, queryFilters)
        responseText = self.rawQuery(searchStringFeatures, queryFilters, "text")

        responseImage = [{"score": item.score, "scoreCustom": item.score*imageW, 
                          "id": item.id, "payload":item.payload} for item in responseImage]
        responseText = [{"score": item.score, "scoreCustom": item.score*textW, 
                         "id": item.id, "payload":item.payload} for item in responseText]

        combinedList = responseImage + responseText
        uniqueItems = {}

        for item in combinedList:
            if item['id'] in uniqueItems:
                # Update if the current score is higher
                if item['score'] > uniqueItems[item['id']]['score']:
                    uniqueItems[item['id']] = item
            else:
                uniqueItems[item['id']] = item
                
        sortedCombined = sorted(uniqueItems.values(), key=lambda x: x['scoreCustom'], reverse=True)
        return sortedCombined
