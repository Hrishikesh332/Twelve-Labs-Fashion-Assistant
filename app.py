import streamlit as st
import time
from twelvelabs import TwelveLabs
import torch
from torchvision import models, transforms
from PIL import Image
import pandas as pd
from urllib.parse import urlparse
import uuid
from dotenv import load_dotenv
import os
from pymilvus import MilvusClient
from pymilvus import connections

load_dotenv()

TWELVELABS_API_KEY = os.getenv('TWELVELABS_API_KEY')
MILVUS_DB_NAME = os.getenv('MILVUS_DB_NAME')
COLLECTION_NAME = os.getenv('COLLECTION_NAME')
MILVUS_HOST = os.getenv('MILVUS_HOST')
MILVUS_PORT = os.getenv('MILVUS_PORT')
URL = os.getenv('URL')
TOKEN = os.getenv('TOKEN')

# Connect to Milvus
connections.connect(
   uri=URL,
   token=TOKEN
)


# Initialize Milvus client
milvus_client = MilvusClient(
    uri=URL,
    token=TOKEN
)

collection_name = COLLECTION_NAME

# Check if collection exists and drop if necessary
if milvus_client.has_collection(collection_name):
    milvus_client.drop_collection(collection_name)

# Create collection with proper schema
milvus_client.create_collection(
    collection_name=collection_name,
    dimension=1024,
    vector_field_name="vector",
    enable_dynamic_field=True
)

# Create index
milvus_client.create_index(
    collection_name=collection_name,
    field_name="vector",
    index_params={
        "metric_type": "COSINE",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 128}
    }
)

# Load collection
milvus_client.load_collection(collection_name)

st.write(f"Collection '{collection_name}' created successfully")
st.write("Hello")

st.markdown("""
    <style>
        .stApp {
            background-color: #f8f9fa;
        }
        .main {
            padding: 2rem;
        }
        .st-emotion-cache-16idsys {
            padding-top: 2rem;
        }
        .st-emotion-cache-1r6slb0 {
            background-color: #ffffff;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .css-1d391kg {
            padding: 1rem;
            border-radius: 10px;
        }
        .stButton>button {
            width: 100%;
            border-radius: 20px;
            background-color: #ff4b6e;
            color: white;
            font-weight: 600;
        }
        .stButton>button:hover {
            background-color: #ff3356;
            color: white;
        }
        .upload-box {
            border: 2px dashed #ccc;
            padding: 20px;
            text-align: center;
            border-radius: 10px;
            margin: 20px 0;
        }
        .results-container {
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin-top: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .sidebar-info {
            margin-top: 20px;
            padding: 10px;
            background-color: #f0f2f6;
            border-radius: 5px;
        }
    </style>
""", unsafe_allow_html=True)


def generate_embedding(video_url):
    try:
        twelvelabs_client = TwelveLabs(api_key="tlk_32YBVAW1GVJHV42ASQ5KB3WEJYW1")
        print(f"Processing video URL: {video_url}")
     
        task = twelvelabs_client.embed.task.create(
            engine_name="Marengo-retrieval-2.6",
            video_url=video_url
        )
        print(f"Created task: id={task.id} engine_name={task.engine_name} status={task.status}")
    
        status = task.wait_for_done(
            sleep_interval=2,
            callback=lambda t: print(f"Status={t.status}")
        )
        print(f"Embedding done: {status}")

        # Get the task result explicitly using the client
        task_result = twelvelabs_client.embed.task.retrieve(task.id)
        print(task_result)
        
        if hasattr(task_result, 'video_embedding') and task_result.video_embedding is not None and task_result.video_embedding.segments is not None:
            embeddings = []
            for segment in task_result.video_embedding.segments:
                embeddings.append({
                    'embedding': segment.embeddings_float,
                    'start_offset_sec': segment.start_offset_sec,
                    'end_offset_sec': segment.end_offset_sec,
                    'embedding_scope': segment.embedding_scope,
                    'video_url': video_url
                })
            return embeddings, task_result, None
        else:
            return None, None, "No embeddings found in task result"
            
    except Exception as e:
        print(f"Error in generate_embedding: {str(e)}")
        return None, None, str(e)
            



class ImageEncoder:
    def __init__(self):
        self.model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        self.projection = torch.nn.Linear(512, 1024)
        self.model.eval()
    
    def encode(self, image):
        if isinstance(image, str):
            img = Image.open(image)
        else:
            img = Image.open(image)
        img = img.convert('RGB')
        
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        img = transform(img).unsqueeze(0)
        
        with torch.no_grad():
            features = self.model(img).squeeze()
            features = self.projection(features)
        
        return features.numpy()


def insert_embeddings(embeddings, video_url):
    data = []
    timestamp = int(time.time())
    
    for i, emb in enumerate(embeddings):
        data.append({
            "id": int(f"{timestamp}{i:03d}"),  
            "vector": emb['embedding'],
            "metadata": {
                "scope": emb['embedding_scope'],
                "start_time": emb['start_offset_sec'],
                "end_time": emb['end_offset_sec'],
                "video_url": video_url
            }
        })

    try:
        insert_result = milvus_client.insert(
            collection_name=COLLECTION_NAME,
            data=data
        )
        return True, len(data)
    except Exception as e:
        return False, str(e)


def search_similar_videos(image, top_k=5):
    encoder = ImageEncoder()
    features = encoder.encode(image)
    
    results = milvus_client.search(
        collection_name=COLLECTION_NAME,
        data=[features],
        output_fields=["metadata"],
        search_params={"metric_type": "COSINE", "params": {"nprobe": 10}},
        limit=top_k
    )
    
    search_results = []
    for result in results[0]:
        metadata = result['entity']['metadata']
        search_results.append({
            'Start Time': f"{metadata['start_time']:.1f}s",
            'End Time': f"{metadata['end_time']:.1f}s",
            'Video URL': metadata['video_url'],
            'Similarity': f"{(1 - abs(result['distance'])) * 100:.2f}%"
        })
    
    return search_results
def main():
    st.title("Video Search and Embedding System")
    
    # Add collection info in sidebar
    try:
        stats = milvus_client.num_entities
        st.sidebar.success(f"Connected to collection: {collection_name}")
        st.sidebar.info(f"Number of video segments: {stats}")
    except Exception as e:
        st.sidebar.error(f"Collection status error: {str(e)}")
    
    tab1, tab2 = st.tabs(["Add Videos", "Search Videos"])
    
    with tab1:
        st.header("Add New Video")
        video_url = st.text_input("Enter Video URL", 
                                placeholder="Enter the URL of your video...")
        
        if st.button("Process Video"):
            with st.spinner("Generating embeddings..."):
                embeddings, task_result, error = generate_embedding(video_url)
                
                if error:
                    st.error(f"Error: {error}")
                else:
                    if embeddings:
                        with st.spinner("Inserting embeddings into database..."):
                            success, result = insert_embeddings(embeddings, video_url)
                            if success:
                                st.success(f"Successfully processed {result} segments from the video!")
                                st.json({
                                    "Total segments": result,
                                    "Sample embedding": {
                                        "Time range": f"{embeddings[0]['start_offset_sec']} - {embeddings[0]['end_offset_sec']} seconds",
                                        "Vector preview": embeddings[0]['embedding'][:5]
                                    }
                                })
                            else:
                                st.error(f"Error inserting embeddings: {result}")
                    else:
                        st.error("No embeddings generated from the video")
    
    with tab2:
        st.header("Search Similar Videos")
        uploaded_file = st.file_uploader(
            "Upload an image to search",
            type=['png', 'jpg', 'jpeg']
        )
        
        if uploaded_file:
            st.image(uploaded_file, caption="Uploaded Image", width=300)
            
            top_k = st.slider("Number of results to show", min_value=1, max_value=20, value=5)
            
            if st.button("Search Similar Videos"):
                with st.spinner("Searching for similar videos..."):
                    results = search_similar_videos(uploaded_file, top_k=top_k)
                    
                    if not results:
                        st.warning("No similar videos found.")
                    else:
                        for idx, result in enumerate(results, 1):
                            with st.expander(f"Match #{idx} - {result['Similarity']} Similar"):
                                st.markdown(f"""
                                #### Time Range
                                {result['Start Time']} - {result['End Time']}
                                
                                #### Video URL
                                {result['Video URL']}
                                """)
                                
                                if st.button(f"Copy URL #{idx}", key=f"copy_{idx}"):
                                    st.code(result['Video URL'])

if __name__ == "__main__":
    main()
