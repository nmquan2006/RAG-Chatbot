import os
import hashlib
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

# Load environment variables from .env file
load_dotenv()

db_path= "vector_db"

embedding_model = "models/embedding-001"
EMBEDDING = GoogleGenerativeAIEmbeddings(model=embedding_model)



def load_vector_database():
    if not os.path.exists(db_path):
        print(f"Error! Can not find folder: {db_path}\n")
        return None
    return Chroma(persist_directory=db_path, embedding_function=EMBEDDING)

def hash_ids(chunks):
    ids = []
    for chunk in chunks:
        content = hashlib.sha256(chunk.page_content.encode('utf-8')).hexdigest()
        page = chunk.metadata.get("page", "0")
        unique_id = f"{content}-p{page}"
        ids.append(unique_id)
    return ids

def add_to_vector_db(chunks, folder_path=db_path):
    
    vector_db = Chroma(persist_directory=folder_path, embedding_function=EMBEDDING)
    
    new_ids = hash_ids(chunks)
    old_data = vector_db.get()
    old_ids = set(old_data['ids'])

    new_chunks = []
    new_ids_to_add = []

    for i, chunk_id in enumerate(new_ids):
        if chunk_id not in old_ids:
            new_chunks.append(chunks[i])
            new_ids_to_add.append(chunk_id)
    
    if new_chunks:
        vector_db.add_documents(documents = new_chunks, ids=new_ids_to_add)
        print(f"Added {len(new_chunks)} new chunks to the vector database.")
    else:
        print("No new chunks to add!")
    
    return vector_db