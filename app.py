import os
import sys
import threading
import time
from typing import Dict, List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import RetrievalQA

from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain.memory import ConversationSummaryBufferMemory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM


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
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100, separators=["\n\n", "\n", ".", " "])
                
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


def embed_docs(documents, embedding_model_name: str) -> None:

    try:
        print("==== Embedding Documents ====")

        embedding_model = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={"device": "cpu"}
        )

        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embedding_model,
            persist_directory="vector_store"
        )

        vector_store.persist()

        print("==== Vector Store Created ====")

        return vector_store
    except Exception as e:
        print(f"Error: {e}")
        return None

def generate_context(question: str, vector_store):
    try:
        #print("==== Generating Context ====")

        retriever = vector_store.as_retriever(search_kwargs={"k": 4})
        docs = retriever.get_relevant_documents(question)
        context = "\n\n".join([doc.page_content for doc in docs])

        #print(f"Retrieved Context: {context}\n\n")

        return context
    except Exception as e:
        print(f"Error: {e}")
        return None

def prompt_template(context: str, question: str):
    try:
        #print("==== Generating Prompt ====")

        template = """
<s>[INST]
You are a language expert. Answer the following question using only the context provided. 
Use the vocabulary and tone found in the context to shape your response — speak as if you were writing from that era. 
Do not add unrelated modern ideas.

Context:
{context}

Question: {question}
</s>
"""
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

        prompt_str = prompt.format(context=context, question=question)

        #print(f"Prompt: {prompt_str}\n\n")
        
        return prompt_str
    except Exception as e:
        print(f"Error: {e}")
        return None


    
def retrieval(vector_store):
    try:
        #print("==== Retrieval ====")

        llm = OllamaLLM(model="llama3")

        memory = ConversationSummaryBufferMemory(llm=llm, memory_key="chat_history", return_messages=True)

        
        rag = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(),
            memory=memory,
        )

        """ rag = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vector_store.as_retriever(),
            memory=memory
        ) """

        return llm
    except Exception as e:
        print(f"Error: {e}")
        return None

def spinner(stop_event):
    spinner_chars = "|/-\\"
    i = 0
    while not stop_event.is_set():
        sys.stdout.write(f"\rThinking... {spinner_chars[i % len(spinner_chars)]}")
        sys.stdout.flush()
        time.sleep(0.1)
        i += 1
    sys.stdout.write("\r" + " " * 50 + "\r")  # Clear the line after it's done

if __name__ == "__main__":

    documents = load_pdfs_from_folder("/Users/bubz/Developer/machine-learning/llm/RAG-LLM/documents")

    vector_store = embed_docs(documents, "all-mpnet-base-v2")

    rag = retrieval(vector_store)

    count = 0
    while True:
        if count == 0:
            question = input("Ask me a question: ")
        else: 
            question = input("Ask me another question: ")

        if question.strip().lower() == "exit":
            print("See ya next time!")
            break

        count += 1


        context = generate_context(question, vector_store)
        prompt = prompt_template(context, question)

        stop_event = threading.Event()
        spin_thread = threading.Thread(target=spinner, args=(stop_event,))
        spin_thread.start()

        try:
            response = rag.invoke(prompt)
            #response = rag.invoke({"query": prompt})
        finally:
            stop_event.set()
            spin_thread.join()

        #print(f"Response: {response['result']}\n\n")
        print(f"Response: {response}\n\n")