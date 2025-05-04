from langchain_community.vectorstores import Chroma

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


    