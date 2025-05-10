from fastapi import FastAPI, Request
from pydantic import BaseModel
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationSummaryBufferMemory
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings

app = FastAPI()

VECTOR_DIR = "vector_store"
EMBEDDING_MODEL = "all-mpnet-base-v2"

class QueryRequest(BaseModel):
    question: str

# Load vector store
embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={"device": "cpu"})
vector_store = Chroma(persist_directory=VECTOR_DIR, embedding_function=embedding)

# Define prompt
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
prompt = PromptTemplate(template=template, input_variables=["context", "question"])

# Create retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 4})

# Create LLM
llm = OllamaLLM(model="llama3")
memory = ConversationSummaryBufferMemory(llm=llm, memory_key="chat_history", return_messages=True)

@app.post("/ask")
async def ask_question(request: QueryRequest):
    # Get relevant documents
    docs = retriever.get_relevant_documents(request.question)
    context = "\n\n".join([doc.page_content for doc in docs])

    # Create prompt
    prompt_str = prompt.format(context=context, question=request.question)

    # Run LLM directly with the prompt
    response = llm.invoke(prompt_str)

    return {"response": response}
