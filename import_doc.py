import os
from typing import Dict, List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.memory import ConversationSummaryMemory
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama

def load_pdfs_from_folder(folder_path) -> List[Dict]:
    """
    Load all PDF files from the specified folder.
    
    Args:
        folder_path (str): Path to the folder containing PDF files
        
    Returns:
        List[Dict]: List of dictionaries containing PDF content and metadata
    """
    try:
        pdf_documents = []
        
        # Ensure the folder exists
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Folder {folder_path} does not exist")
        
        # Get all PDF files in the folder
        pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]
        
        for pdf_file in pdf_files:
            file_path = os.path.join(folder_path, pdf_file)
            print(f"file_path: {file_path}")
            try:
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=30, length_function=len)
                
                document = PyPDFLoader(file_path).load_and_split(text_splitter=text_splitter)
                
                
                pdf_documents.extend(document)  

            except Exception as e:
                print(f"Error processing {pdf_file}: {str(e)}")

        for i, chunks in enumerate(pdf_documents):
            if chunks:
                print(f"---- Document {i+1} ----")
                print(chunks.page_content[:500])
                print(f"---- Document {i+1} ----")
                print("\n")
    
        return pdf_documents
    except Exception as e:
        print(f"Error: {e}")
        return None


if __name__ == "__main__":
    # Example usage
    documents = load_pdfs_from_folder("/Users/bubz/Developer/machine-learning/llm/RAG-LLM/documents")

    for i, chunks in enumerate(documents):
        if chunks:
            print(f"---- Document {i+1} ----")
            print(chunks.page_content[:500])
            print(f"---- Document {i+1} ----")
            print("\n")

