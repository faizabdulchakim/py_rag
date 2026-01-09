from fastapi import FastAPI
from pydantic import BaseModel
import os
import tiktoken

from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOC_PATH = os.path.join(BASE_DIR, "data", "docs.txt")

loader = TextLoader(DOC_PATH, encoding="utf-8")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50
)
chunks = splitter.split_documents(documents)

embedding_model = OllamaEmbeddings(
    model="nomic-embed-text"
)

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="./chroma_db"
)

llm = OllamaLLM(
    model="phi4:latest",
    temperature=0
)

def distance_to_similarity(distance: float) -> float:
    return round(1 / (1 + distance), 4)

def estimate_tokens(text: str) -> int:
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))

class Query(BaseModel):
    question: str
    top_k: int = 3

@app.post("/rag")
def rag(query: Query):

    results = vectorstore.similarity_search_with_score(
        query.question,
        k=query.top_k
    )

    retrieved_docs = []
    context_blocks = []

    for doc, distance in results:
        retrieved_docs.append({
            "content": doc.page_content,
            "similarity_score": distance_to_similarity(distance),
            "metadata": doc.metadata
        })
        context_blocks.append(doc.page_content)

    context = "\n\n".join(context_blocks)

    prompt = f"""
You are a helpful assistant.
Answer ONLY using the context below.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{query.question}

Answer:
""".strip()

    answer = llm.invoke(prompt)

    prompt_tokens = estimate_tokens(prompt)
    completion_tokens = estimate_tokens(answer)

    token_usage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens
    }

    return {
        "question": query.question,
        "answer": answer,
        "retrieved_documents": retrieved_docs,
        "prompt_used": prompt,
        "token_usage_estimate": token_usage
    }
