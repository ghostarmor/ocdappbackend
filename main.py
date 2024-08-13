import os
from typing import List, Tuple
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import DeepInfra
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Load CSV data
df = pd.read_csv("btsldataset.csv")
questions = df["Question"].tolist()
answers = df["Response"].tolist()

# Initialize embeddings
embeddings = HuggingFaceEmbeddings()

# Create vector store
vectorstore = FAISS.from_texts(questions, embeddings)


# Initialize Llama 3.1 LLM from DeepInfra
llm = DeepInfra(model_id="meta-llama/Meta-Llama-3.1-8B-Instruct", deepinfra_api_token=os.getenv("DEEPINFRA_API_TOKEN"))
llm.model_kwargs = {
    "temperature": 0.7,
    "repetition_penalty": 1.0,
    "max_new_tokens": 250,
    "top_p": 0.9,
}


# Function to retrieve most relevant question and its corresponding answer
def get_relevant_qa(query: str):
    relevant_docs = vectorstore.similarity_search(query, k=1)
    if relevant_docs:
        relevant_question = relevant_docs[0].page_content
        relevant_answer = answers[questions.index(relevant_question)]
        return relevant_question, relevant_answer
    return "", ""

# Create prompt template 
prompt_template = """Adapt this similar question + answer to the user's question and give them a fitting answer. Provide a concise and direct answer without mentioning the context or explaining your thought process. If you don't know the answer, just say that you don't know, don't try to make up an answer.

User Question: {user_question}

Relevant Question from Database: {relevant_question}

Corresponding Answer from Database: {relevant_answer}

Based on the above context, please provide an answer to the user's question:

Answer: """

PROMPT = PromptTemplate(
    template=prompt_template, 
    input_variables=["user_question", "relevant_question", "relevant_answer"]
)

# Create LLM chain
llm_chain = LLMChain(llm=llm, prompt=PROMPT)

class Query(BaseModel):
    question: str

@app.post("/ask")
def ask_question(query: Query):
    relevant_question, relevant_answer = get_relevant_qa(query.question)
    if relevant_answer == "" and relevant_question == "":
        return {
            "answer": "I don't know how to answer that.",
            "relevant_question": "N/a",
            "relevant_answer": "N/a"
        }
    result = llm_chain.run(
        user_question=query.question,
        relevant_question=relevant_question,
        relevant_answer=relevant_answer
    )
    return {
        "answer": result,
        "relevant_question": relevant_question,
        "relevant_answer": relevant_answer
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)