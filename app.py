import os
from typing import Dict, List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import RetrievalQA
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
        print("==== Generating Context ====")

        retriever = vector_store.as_retriever(search_kwargs={"k": 4})
        docs = retriever.get_relevant_documents(question)
        context = "\n\n".join([doc.page_content for doc in docs])

        print(f"Context: {context}\n\n")

        return context
    except Exception as e:
        print(f"Error: {e}")
        return None

def prompt_template(context: str, question: str):
    try:
        print("==== Generating Prompt ====")

        template = """
        <s>[INST]
        Given the following context, answer the question.
        {context}
        Question: {question}
        </s>
        """
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

        prompt_str = prompt.format(context=context, question=question)

        print(f"Prompt: {prompt_str}\n\n")
        
        return prompt
    except Exception as e:
        print(f"Error: {e}")
        return None


    
def retrieval(question, prompt_template, vector_store):
    try:
        print("==== Retrieval ====")

        llm = OllamaLLM(model="mistral")

        memory = ConversationSummaryBufferMemory(llm=llm, memory_key="chat_history", return_messages=True)

        rag = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(),
            memory=memory,
            chain_type_kwargs={"prompt": prompt_template}
        )

        print(f"Print Chain: {prompt.template}")

        response = rag.invoke({"query": question})
        return response
    except Exception as e:
        print(f"Error: {e}")
        return None


if __name__ == "__main__":

    question = """Rewrite the following modern-day story using the slang and expressions from the provided context. Imagine it's being told by two friends from 1910s London:

"Man, today has been rough. I missed the bus, spilled coffee all over my shirt, and my boss chewed me out in front of the whole office. After all that, I just wanted to grab a burger and chill, but guess what — I forgot my wallet. Again. Now I'm just wandering around the city, hoping to find some peace before heading home."

Keep it conversational and reflective, like someone venting to a close mate in the pub after a long day."""
    documents = load_pdfs_from_folder("/Users/bubz/Developer/machine-learning/llm/RAG-LLM/documents")

    vector_store = embed_docs(documents, "all-mpnet-base-v2")

    context = generate_context(question, vector_store)

    prompt = prompt_template(context, question)

    response = retrieval(question, prompt, vector_store)

    print(f"Response: {response}\n\n")