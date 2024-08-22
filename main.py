import os
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.chat_models import ChatDeepInfra
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, PromptTemplate
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Load YT transcript data
loader = DirectoryLoader("./mfreemantranscripts", glob="**/*.txt", loader_cls=TextLoader)
docs = loader.load()

print(len(docs))

# Split YT transcript docs
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)

print(len(all_splits))

# Initialize embeddings
model_name = "BAAI/bge-small-en"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)

# Create vector store for YT transcript docs
yt_transcript_vectorstore = FAISS.from_documents(all_splits, embeddings)

# Load CSV data
df = pd.read_csv("btsldataset.csv")
questions = df["Question"].tolist()
answers = df["Response"].tolist()

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

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Function to retrieve most relevant questions and its corresponding answers using both semantic and keyword search
def get_relevant_qa(query: str):
    # Semantic search
    relevant_docs = vectorstore.similarity_search(query, k=3)
    relevant_questions = [q.page_content for q in relevant_docs]
    #relevant_answers = [answers[questions.index(q)] for q in relevant_questions]
    
    # Keyword search
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(questions)
    query_vec = tfidf.transform([query])
    cosine_similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    keyword_relevant_indices = cosine_similarities.argsort()[-3:][::-1]
    
    keyword_relevant_questions = [questions[i] for i in keyword_relevant_indices]
    #keyword_relevant_answers = [answers[i] for i in keyword_relevant_indices]
    
    # Combine and deduplicate results
    combined_questions = list(dict.fromkeys(relevant_questions + keyword_relevant_questions))
    combined_answers = [answers[questions.index(q)] for q in combined_questions]
    
    return combined_questions[:3], combined_answers[:3]  # Return top 3 combined results

def get_relevant_docs(query: str, top_k: int):
    relevant_docs = yt_transcript_vectorstore.similarity_search(query, k=top_k)
    if relevant_docs:
        return format_docs(relevant_docs)
    return "N/a"

agent1_prompt_template = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "You are an AI assistant specialized in mental health and OCD. If the question is not related to mental health, say that you cannot answer it. Otherwise, provide a direct, concise answer to the user's question based on the relevant information provided. The context provided will include relevant questions and answers that you can refer to and also relevant mental fitness information you can use when generating your answer. Do not include any analysis steps, thought processes, or explanations of your answer. Just give the final, helpful response and ensure it makes sense in the context of the user's question."
    ),
    ("human", "User question: {user_question}\n\nRelevant general information:\n{context}\n\nRelevant question/answer information:\n{qa_context}\n\nConcise answer:")
])

reviewer_agent_prompt_template = PromptTemplate.from_template(
    '''
    Your job is to evaluate if the response by Agent 1 makes sense relative to the user's question. You must evaluate if the response has any parts that are not relevant or useful to answer the user's question. If it does make sense, reply "Yes" and nothing else. If it does not make complete sense, reply with the edited response that makes more sense.
    Question: {question}
    Agent 1's Response: {response}
    Your answer: 
    '''
)

class Query(BaseModel):
    question: str

@app.post("/ask")
def ask_question(query: Query):
    top_k_docs = 5
    
    relevant_questions, relevant_answers = get_relevant_qa(query.question)
    top_k_docs -= len(relevant_questions)
    
    context = get_relevant_docs(query=query.question, top_k=top_k_docs)
    
    qa_context = ""
    for i, (q, a) in enumerate(zip(relevant_questions, relevant_answers), 1):
        qa_context += f'Relevant question retrieved from database - #{i}: {q}\nCorresponding answer retrieved from database - #{i}: {a}\n'
    
    agent1_prompt = agent1_prompt_template.invoke({"user_question":query.question, "context": context , "qa_context": qa_context})
    result = llm.invoke(agent1_prompt)
    final_response = result.content
    metadata1 = result.response_metadata
    
    # review_agent_prompt = reviewer_agent_prompt_template.invoke({"question": query.question, "response": result.content})
    # review = llm.invoke(review_agent_prompt)
    # if review.content.strip().lower() != 'yes':
    #     print(f'Reviewer made changes. Original response was {final_response}')
    #     final_response = review.content
    
    # metadata2 = review.response_metadata
    
    return {
        "answer": final_response,
        "relevant_questions": relevant_questions,
        "relevant_answers": relevant_answers,
        "relevant_docs": context,
        "input_tokens": metadata1['token_usage']['prompt_tokens'], #+ metadata2['token_usage']['prompt_tokens'],
        "output_tokens": metadata1['token_usage']['completion_tokens'], #+ metadata2['token_usage']['completion_tokens'],
        "estimated_cost": metadata1['token_usage']['estimated_cost'], #+ metadata2['token_usage']['estimated_cost']
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)