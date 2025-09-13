from qdrant_client import QdrantClient
from qdrant_client.http import models
from openai import OpenAI
import uuid
import streamlit as st

def init_qdrant():
    client = QdrantClient(
        url=st.secrets["QDRANT_URL"],
        api_key=st.secrets["QDRANT_API_KEY"]
    )
    # upewniamy się, że kolekcja istnieje
    client.recreate_collection(
        collection_name="conversations",
        vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE),
    )
    return client

def save_to_qdrant(user_text, assistant_text, conv_id, client):
    openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    combined = user_text + " " + assistant_text
    embedding = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=combined
    ).data[0].embedding

    client.upsert(
        collection_name="conversations",
        points=[models.PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload={"conversation_id": conv_id, "user": user_text, "assistant": assistant_text}
        )]
    )
