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
from openai import OpenAI

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

# Create an OpenAI client with your deepinfra token and endpoint
openai = OpenAI(
    api_key=os.getenv("DEEPINFRA_API_TOKEN"),
    base_url="https://api.deepinfra.com/v1/openai",
)


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
    relevant_docs = vectorstore.similarity_search(query, k=3)
    if relevant_docs:
        relevant_questions = [q.page_content for q in relevant_docs]
        relevant_answers = [answers[questions.index(q)] for q in relevant_questions]
        return relevant_questions, relevant_answers
    return [],[]

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
    relevant_questions, relevant_answers = get_relevant_qa(query.question)
    if relevant_answers == [] and relevant_questions == []:
        return {
            "answer": "I don't know how to answer that.",
            "relevant_question": "N/a",
            "relevant_answer": "N/a"
        }
    
    context = ""
    for i in range(len(relevant_questions)):
        context += f'Relevant question retrieved from database #{i+1}: {relevant_questions[i]}\nCorresponding answer retrieved from database #{i+1}: {relevant_answers[i]}\n'
    
    chat_completion = openai.chat.completions.create(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct",
        messages=[
                {"role": "system", "content": "Your task is to state the relation (if there is one) between the user's question, and the relevant questions and corresponding answers provided to you as context. Be direct and concise, and state how each piece of the context provided can be adapted to answer the user's question regarding mental health and OCD."},
                {"role": "user", "content": f"""
                 User question: {query.question}
                 {context}
                 """},
            ],
    )
    
    result = chat_completion.choices[0].message.content
    token_cost = chat_completion.usage.prompt_tokens, chat_completion.usage.completion_tokens
    
    
    # result = llm_chain.run(
    #     user_question=query.question,
    #     relevant_question=relevant_question,
    #     relevant_answer=relevant_answer
    # )
    return {
        "answer": result,
        "relevant_questions": relevant_questions,
        "relevant_answers": relevant_answers,
        "usage_tokens": token_cost
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)