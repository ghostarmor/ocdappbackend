import os
from typing import List, Tuple
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatDeepInfra
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate

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
llm = ChatDeepInfra(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    deepinfra_api_token=os.getenv("DEEPINFRA_API_TOKEN")
)
llm.model_kwargs = {
    "temperature": 0.7,
    "repetition_penalty": 1.0,
    "max_new_tokens": 250,
    "top_p": 0.9,
}

# Function to retrieve most relevant question and its corresponding answer
def get_relevant_qa(query: str):
    relevant_docs = vectorstore.similarity_search(query, k=3)
    if relevant_docs:
        relevant_questions = [q.page_content for q in relevant_docs]
        relevant_answers = [answers[questions.index(q)] for q in relevant_questions]
        return relevant_questions, relevant_answers
    return [], []


agent1_prompt_template = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "You are an AI assistant specialized in mental health and OCD. Provide a direct, concise answer to the user's question based on the relevant information provided. Do not include any analysis steps, thought processes, or explanations of your answer. Just give the final, helpful response."
    ),
    ("human", "User question: {user_question}\n\nRelevant information:\n{context}\n\nConcise answer:")
])





# Create LLM chain
#llm_chain = LLMChain(llm=llm, prompt=prompt)

class Query(BaseModel):
    question: str

@app.post("/ask")
def ask_question(query: Query):
    relevant_questions, relevant_answers = get_relevant_qa(query.question)
    if not relevant_questions:
        return {
            "answer": "I don't know how to answer that.",
            "relevant_question": "N/a",
            "relevant_answer": "N/a"
        }
    
    context = ""
    for i, (q, a) in enumerate(zip(relevant_questions, relevant_answers), 1):
        context += f'Relevant question retrieved from database #{i}: {q}\nCorresponding answer retrieved from database #{i}: {a}\n'
    
    agent1_prompt = agent1_prompt_template.invoke({"user_question":query.question, "context": context})
    initial_result = llm.invoke(agent1_prompt)
    
    
    
    # Note: LangChain doesn't provide token usage for DeepInfra LLMs out of the box
    # You might need to implement a custom callback to track token usage if needed
    
    return {
        "answer": initial_result.content,
        "relevant_questions": relevant_questions,
        "relevant_answers": relevant_answers,
        # "usage_tokens": Not available through LangChain for DeepInfra
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)