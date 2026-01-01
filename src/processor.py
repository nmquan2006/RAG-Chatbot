import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def process_documents(data_path = "data/"):

    if not os.path.exists(data_path):
        print(f"Errol! Cannot find data path :{data_path}")
        return None
    
    documents_loader = DirectoryLoader(data_path, glob = "**/*.pdf", loader_cls = PyPDFLoader)

    try:
        documents = documents_loader.load()
    except Exception as e:
        print(f"Errol! Failed to load documents from {data_path}. Error: {e}")
        return None
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 700,
        chunk_overlap = 100,
        separators = ["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    return chunks

# Test the function (run only when this script is executed directly)
if __name__ == "__main__":
    result = process_documents()
    if result:
        print(result[2].page_content)