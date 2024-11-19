import streamlit as st
import time
from twelvelabs import TwelveLabs
from pymilvus import MilvusClient
import torch
from torchvision import models, transforms
from PIL import Image
import pandas as pd
from urllib.parse import urlparse
import uuid
from dotenv import load_dotenv
import os


load_dotenv()


TWELVELABS_API_KEY = os.getenv('TWELVELABS_API_KEY')
MILVUS_DB_NAME = os.getenv('MILVUS_DB_NAME')
COLLECTION_NAME = os.getenv('COLLECTION_NAME')
MILVUS_HOST = os.getenv('MILVUS_HOST')
MILVUS_PORT = os.getenv('MILVUS_PORT')

milvus_client = MilvusClient(
    uri=f"http://{MILVUS_HOST}:{MILVUS_PORT}",
    db_name=MILVUS_DB_NAME
)

st.set_page_config(
    page_title="Fashion AI Assistant",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

class ModifiedResNet34:
    def __init__(self):
        self.model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        self.projection = torch.nn.Linear(512, 1024)
        self.model.eval()
    
    def __call__(self, image):
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

def get_direct_drive_link(sharing_url):
    file_id = None
    if 'drive.google.com' in sharing_url:
        if '/file/d/' in sharing_url:
            file_id = sharing_url.split('/file/d/')[1].split('/')[0]
        elif 'id=' in sharing_url:
            file_id = sharing_url.split('id=')[1].split('&')[0]
    
    if not file_id:
        raise ValueError("Could not extract file ID from Google Drive URL")
    
    return f"https://drive.google.com/uc?export=download&id={file_id}"

def generate_embedding(drive_url):
    try:
        direct_url = get_direct_drive_link(drive_url)
        twelvelabs_client = TwelveLabs(api_key=TWELVELABS_API_KEY)

        task = twelvelabs_client.embed.task.create(
            engine_name="Marengo-retrieval-2.6",
            video_url=direct_url
        )

        with st.spinner("Processing video..."):
            status = task.wait_for_done(sleep_interval=2)
            task_result = twelvelabs_client.embed.task.retrieve(task.id)

        embeddings = []
        for segment in task_result.video_embedding.segments:
            embeddings.append({
                'embedding': segment.embeddings_float,
                'start_offset_sec': segment.start_offset_sec,
                'end_offset_sec': segment.end_offset_sec,
                'embedding_scope': segment.embedding_scope,
                'video_url': drive_url
            })

        return embeddings, task_result, None
    except Exception as e:
        return None, None, str(e)

def search_similar_products(image, top_k=5):
    extractor = ModifiedResNet34()
    features = extractor(image)
    
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
            'Clip ID': result['id'],
            'Start Time': f"{metadata['start_time']:.1f}s",
            'End Time': f"{metadata['end_offset_sec']:.1f}s",
            'Video URL': metadata['video_url'],
            'Similarity': f"{(1 - abs(result['distance'])) * 100:.2f}%"
        })
    
    return search_results

def check_env_variables():
    """Check if all required environment variables are set"""
    required_vars = [
        'TWELVELABS_API_KEY',
        'MILVUS_DB_NAME',
        'COLLECTION_NAME',
        'MILVUS_HOST',
        'MILVUS_PORT'
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        st.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        st.stop()

def main():

    check_env_variables()
    

    with st.sidebar:
        st.image("https://via.placeholder.com/150?text=Fashion+AI", width=150)
        st.title("Navigation")
        selected = st.radio(
            "",
            ["🎥 Add Videos", "🤖 Chat Assistant"],
            format_func=lambda x: x.split(" ")[1]
        )
      
        with st.expander("ℹ️ System Info"):
            st.info(f"""
            - Database: {MILVUS_DB_NAME}
            - Collection: {COLLECTION_NAME}
            - Host: {MILVUS_HOST}
            - Port: {MILVUS_PORT}
            """)
    
    if selected == "🎥 Add Videos":
        st.title("📚 Fashion Video Database")
        st.subheader("Add new fashion videos to the database")
        
        with st.form("video_form", clear_on_submit=True):
            col1, col2 = st.columns([3, 1])
            with col1:
                video_url = st.text_input("🔗 Enter Google Drive Video URL", 
                                        placeholder="https://drive.google.com/file/d/...")
            with col2:
                submitted = st.form_submit_button("➕ Add Video")
            
            if submitted and video_url:
                embeddings, task_result, error = generate_embedding(video_url)
                
                if error:
                    st.error(f"❌ Error: {error}")
                else:
                    current_count = milvus_client.get_collection_stats(COLLECTION_NAME)['row_count']
                    insert_result, inserted_count = insert_embeddings(
                        milvus_client,
                        COLLECTION_NAME,
                        embeddings,
                        start_id=current_count
                    )
                    st.success(f"✅ Successfully added {inserted_count} segments!")
                    
                    # Display progress
                    st.metric("Total Videos", 
                            milvus_client.get_collection_stats(COLLECTION_NAME)['row_count'])
    
    else:  # Chat Assistant page
        st.title("👗 Fashion Product Assistant")
        st.subheader("Upload a product image to find similar items")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            uploaded_file = st.file_uploader(
                "Drop your image here",
                type=['png', 'jpg', 'jpeg'],
                help="Supported formats: PNG, JPG, JPEG"
            )
            
            if uploaded_file:
                st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
                if st.button("🔍 Find Similar Products", use_container_width=True):
                    with st.spinner("🕵️‍♀️ Searching for similar products..."):
                        results = search_similar_products(uploaded_file)
                        
                        with col2:
                            st.markdown("### 🎯 Similar Products Found")
                            for idx, result in enumerate(results, 1):
                                with st.container():
                                    st.markdown(f"""
                                    #### Match #{idx} - {result['Similarity']} Similar
                                    - 🎬 Time: {result['Start Time']} - {result['End Time']}
                                    - 🔗 [Watch Video]({result['Video URL']})
                                    """)
                                    st.divider()

if __name__ == "__main__":
    main()