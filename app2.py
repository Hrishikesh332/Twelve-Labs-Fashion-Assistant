from pymilvus import MilvusClient
import streamlit as st

milvus_client = MilvusClient("milvus_twelvelabs_demo3.db")

collection_name = "twelvelabs_collection_dress2"

if milvus_client.has_collection(collection_name=collection_name):
    milvus_client.drop_collection(collection_name=collection_name)


milvus_client.create_collection(
    collection_name=collection_name,
    dimension=1024
)

st.write(f"Collection '{collection_name}' created successfully")
