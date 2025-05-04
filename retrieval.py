from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import RetrievalQA

def retrieval(prompt, vector_store):
    try:
        print("==== Retrieval ====")

        llm = Ollama(model="mistral")

        memory = ConversationSummaryBufferMemory(llm=llm, memory_key="chat_history")

        rag = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vector_store.as_retriever(),
            memory=memory,
            chain_type_kwargs={"prompt": prompt, "verbose": True}
        )

        print(f"Print Chain: {prompt.template}")

        response = rag.run({"query": prompt})
        return response
    except Exception as e:
        print(f"Error: {e}")
        return None 

