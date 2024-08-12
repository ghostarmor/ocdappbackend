import os
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import DeepInfra
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnablePassthrough
# from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Load CSV data
df = pd.read_csv("btsldataset.csv")
questions = df["Question"].tolist()

# Initialize embeddings
embeddings = HuggingFaceEmbeddings()

# Create vector store
vectorstore = FAISS.from_texts(questions, embeddings)

# Initialize Llama 3.1 LLM from DeepInfra
llm = DeepInfra(model_id="meta-llama/Meta-Llama-3.1-8B-Instruct", deepinfra_api_token=os.getenv("DEEPINFRA_API_TOKEN"))

# Create prompt template
prompt_template = """Use the following piece of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}

Question: {question}

Answer: """

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

# Create retrieval chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
    return_source_documents=False,
    chain_type_kwargs={"prompt": PROMPT}
)

class Query(BaseModel):
    question: str

@app.post("/ask")
def ask_question(query: Query):
    result = qa_chain({"query": query.question})
    return {"answer": result["result"]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)