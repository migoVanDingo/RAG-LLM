from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from import_doc import load_pdfs_from_folder

def embeddings(documents, embedding_model_name: str) -> None:
    #embedding_model_name = "all-mpnet-base-v2"

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


if __name__ == "__main__":
    documents = load_pdfs_from_folder("/Users/bubz/Developer/machine-learning/llm/RAG-LLM/documents")
    embeddings(documents, "all-mpnet-base-v2")
