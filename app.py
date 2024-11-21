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
from pymilvus import (
   FieldSchema, DataType, 
   CollectionSchema, Collection,
   utility
)

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

# Define fields for schema
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=1024),
]

# Create schema with dynamic fields for metadata
schema = CollectionSchema(
    fields=fields,
    enable_dynamic_field=True
)

# Check if collection exists
if utility.has_collection(COLLECTION_NAME):
    # If exists, just load the existing collection
    collection = Collection(COLLECTION_NAME)
    print(f"Using existing collection: {COLLECTION_NAME}")
else:
    # If doesn't exist, create new collection
    collection = Collection(COLLECTION_NAME, schema)
    print(f"Created new collection: {COLLECTION_NAME}")
    
    # Create index for new collection
    if not collection.has_index():
        collection.create_index(
            field_name="vector",
            index_params={
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128}
            }
        )
        print("Created index for the new collection")

# Load collection for searching
collection.load()

# Set the milvus_client to the collection
milvus_client = collection

# st.write(f"Connected to collection: {COLLECTION_NAME}")


# # Initialize Milvus client
# milvus_client = MilvusClient(
#     uri=URL,
#     token=TOKEN
# )

# collection_name = COLLECTION_NAME

# # Check if collection exists and drop if necessary
# if milvus_client.has_collection(collection_name):
#     milvus_client.drop_collection(collection_name)

# # Create collection with proper schema
# milvus_client.create_collection(
#     collection_name=collection_name,
#     dimension=1024,
#     vector_field_name="vector",
#     enable_dynamic_field=True
# )

# # Create index
# milvus_client.create_index(
#     collection_name=collection_name,
#     field_name="vector",
#     index_params={
#         "metric_type": "COSINE",
#         "index_type": "IVF_FLAT",
#         "params": {"nlist": 128}
#     }
# )

# # Load collection
# milvus_client.load_collection(collection_name)

# st.write(f"Collection '{COLLECTION_NAME}' created successfully")
# st.write("Hello")

# Clean, professional CSS styling
st.markdown("""
    <style>

          [data-testid="stAppViewContainer"] {
             background-image: url("https://img.freepik.com/free-photo/vivid-blurred-colorful-wallpaper-background_58702-3883.jpg");
             background-size: cover;
         }
         [data-testid="stHeader"] {
             background-color: rgba(0,0,0,0);
         }
         [data-testid="stToolbar"] {
             right: 2rem;
             background-image: url("");
             background-size: cover;
         }
        /* Base styling */
        .stApp {
            background-color: #f8f9fa;
        }
        
        /* Typography */
        h1 {
            color: #1a1f36;
            font-size: 2rem !important;
            font-weight: 600 !important;
            margin-bottom: 1.5rem !important;
            padding: 1rem 0;
        }
        
        h2, h3 {
            color: #1a1f36;
            font-weight: 500;
            margin-bottom: 1rem;
        }
        
        /* Container styling */
        .content-section {
            background: white;
            padding: 1.5rem;
            border-radius: 6px;
            border: 1px solid #e5e7eb;
            margin-bottom: 1.5rem;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        }
        
        /* Input styling */
        .stTextInput input {
            border: 1px solid #e5e7eb;
            border-radius: 4px;
            padding: 0.5rem;
            font-size: 0.875rem;
        }
        
        .stTextInput input:focus {
            border-color: #2563eb;
            box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.1);
        }
        
        /* Button styling */
        .stButton > button {
            background-color: #2563eb;
            color: white;
            font-weight: 500;
            padding: 0.5rem 1rem;
            border-radius: 4px;
            border: none;
            transition: background-color 0.2s;
        }
        
        .stButton > button:hover {
            background-color: #1d4ed8;
        }
        
        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 1px;
            background-color: #f1f5f9;
            padding: 0.5rem;
            border-radius: 4px;
        }
        
        .stTabs [data-baseweb="tab"] {
            background-color: transparent;
            color: #475569;
            font-weight: 500;
            padding: 0.5rem 1rem;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: #2563eb !important;
            color: white !important;
        }
        
        /* File uploader styling */
        .uploadedFile {
            background-color: #f8fafc;
            padding: 1rem;
            border-radius: 4px;
            border: 1px dashed #cbd5e1;
        }
        
        /* Status messages */
        .status-message {
            padding: 0.75rem;
            border-radius: 4px;
            margin: 0.75rem 0;
        }
        
        .success-message {
            background-color: #f0fdf4;
            border-left: 4px solid #22c55e;
            color: #166534;
        }
        
        .error-message {
            background-color: #fef2f2;
            border-left: 4px solid #ef4444;
            color: #991b1b;
        }
        
        /* Sidebar styling */
        .css-1d391kg {
            background-color: white;
            border-right: 1px solid #e5e7eb;
        }
        
        /* Expander styling */
        .streamlit-expanderHeader {
            background-color: #f8fafc;
            border: 1px solid #e5e7eb;
            border-radius: 4px;
        }
        
        /* Progress bar */
        .stProgress > div > div {
            background-color: #2563eb;
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
        # Modified insert call for Collection object
        insert_result = milvus_client.insert(data)
        # Force flush to ensure data is persisted
        milvus_client.flush()
        return True, len(data)
    except Exception as e:
        return False, str(e)


def search_similar_videos(image, top_k=5):
    encoder = ImageEncoder()
    features = encoder.encode(image)
    

    results = milvus_client.search(
        data=[features],
        anns_field="vector",
        param={"metric_type": "COSINE", "params": {"nprobe": 10}},
        limit=top_k,
        output_fields=["metadata"]
    )
    
    search_results = []
    for hits in results:
        for hit in hits:
            metadata = hit.entity.get('metadata')
            if metadata:
                search_results.append({
                    'Start Time': f"{metadata['start_time']:.1f}s",
                    'End Time': f"{metadata['end_time']:.1f}s",
                    'Video URL': metadata['video_url'],
                    'Similarity': f"{(1 - float(hit.distance)) * 100:.2f}%"
                })
    
    return search_results


def format_time_for_url(seconds):
    return f"{int(float(seconds))}"

def get_video_id_from_url(url):

    parsed_url = urlparse(url)

    # Vimeo
    if 'vimeo.com' in url:
        return parsed_url.path[1:], 'vimeo'
    
    # Direct video URL
    elif url.endswith(('.mp4', '.webm', '.ogg')):
        return url, 'direct'
    
    return None, None

def create_video_embed(video_url, start_time, end_time):
    video_id, platform = get_video_id_from_url(video_url)
    start_seconds = format_time_for_url(start_time)
    
    if platform == 'vimeo':
        return f"""
            <iframe 
                width="100%" 
                height="315" 
                src="https://player.vimeo.com/video/{video_id}#t={start_seconds}s"
                frameborder="0" 
                allow="autoplay; fullscreen; picture-in-picture" 
                allowfullscreen>
            </iframe>
        """
    elif platform == 'direct':
        return f"""
            <video 
                width="100%" 
                height="315" 
                controls 
                autoplay
                id="video-player">
                <source src="{video_url}" type="video/mp4">
                Your browser does not support the video tag.
            </video>
            <script>
                document.getElementById('video-player').addEventListener('loadedmetadata', function() {{
                    this.currentTime = {start_time};
                }});
            </script>
        """
    else:
        return f"<p>Unable to embed video from URL: {video_url}</p>"

        
def main():
    st.title("Video Search and Embedding System")
    
    # Sidebar with system status
    with st.sidebar:
        st.subheader("System Status")
        try:
            stats = milvus_client.num_entities
            st.success(f"Connected to: {COLLECTION_NAME}")
            st.info(f"Total Video Segments: {stats:,}")
        except Exception as e:
            st.error(f"Connection Error: {str(e)}")
    
    # Main content
    tabs = st.tabs(["Add Videos", "Search Videos"])
    
    with tabs[0]:
        st.header("Add New Video")
        with st.container():
            st.markdown('<div class="content-section">', unsafe_allow_html=True)
            
            video_url = st.text_input(
                "Video URL",
                placeholder="Enter the URL of your video file",
                help="Provide the complete URL of the video you want to process"
            )
            
            if st.button("Process Video", use_container_width=True):
                with st.spinner("Processing video..."):
                    embeddings, task_result, error = generate_embedding(video_url)
                    
                    if error:
                        st.error(f"Error: {error}")
                    elif embeddings:
                        with st.spinner("Storing embeddings..."):
                            success, result = insert_embeddings(embeddings, video_url)
                            if success:
                                st.success(f"Successfully processed {result} segments")
                                
                                with st.expander("View Processing Details"):
                                    st.json({
                                        "Segments processed": result,
                                        "Sample embedding": {
                                            "Time range": f"{embeddings[0]['start_offset_sec']} - {embeddings[0]['end_offset_sec']} seconds",
                                            "Vector preview": embeddings[0]['embedding'][:5]
                                        }
                                    })
                            else:
                                st.error(f"Error inserting embeddings: {result}")
                    else:
                        st.error("No embeddings generated from the video")
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    with tabs[1]:
        st.header("Search Similar Videos")
        with st.container():
            st.markdown('<div class="content-section">', unsafe_allow_html=True)
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                uploaded_file = st.file_uploader(
                    "Upload Image",
                    type=['png', 'jpg', 'jpeg'],
                    help="Select an image to find similar video segments"
                )
                
                if uploaded_file:
                    st.image(uploaded_file, caption="Query Image", use_column_width=True)
            
            with col2:
                if uploaded_file:
                    st.subheader("Search Parameters")
                    top_k = st.slider(
                        "Number of results",
                        min_value=1,
                        max_value=20,
                        value=2,
                        help="Select the number of similar videos to retrieve"
                    )
                    
                    if st.button("Search", use_container_width=True):
                        with st.spinner("Searching for similar videos..."):
                            results = search_similar_videos(uploaded_file, top_k=top_k)
                            
                            if not results:
                                st.warning("No similar videos found")
                            else:
                                st.subheader("Results")
                                for idx, result in enumerate(results, 1):
                                    with st.expander(f"Match #{idx} - Similarity: {result['Similarity']}", expanded=(idx==1)):
                                        # Extract start and end times
                                        start_time = float(result['Start Time'].replace('s', ''))
                                        end_time = float(result['End Time'].replace('s', ''))
                                        
                                        # Create two columns for video and details
                                        video_col, details_col = st.columns([2, 1])
                                        
                                        with video_col:
                                            st.markdown("#### Video Segment")
                                            # Embed video starting at the specific time
                                            video_embed = create_video_embed(
                                                result['Video URL'],
                                                start_time,
                                                end_time
                                            )
                                            st.markdown(video_embed, unsafe_allow_html=True)
                                        
                                        with details_col:
                                            st.markdown("#### Details")
                                            st.markdown(f"""
                                                🕒 **Time Range**  
                                                {result['Start Time']} - {result['End Time']}
                                                
                                                📊 **Similarity Score**  
                                                {result['Similarity']}
                                                
                                                🔗 **Video URL**
                                            """)
                                            if st.button("📋 Copy URL", key=f"copy_{idx}"):
                                                st.code(result['Video URL'])
            
            st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
