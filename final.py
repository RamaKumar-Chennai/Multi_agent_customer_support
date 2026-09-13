#Import the required libraries

from langchain_core.documents import Document
#from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import FAISS

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import gradio as gr
from langchain_community.llms import Ollama

import pandas as pd

from langchain_core.tools import tool
import joblib
import time
from sklearn.feature_extraction.text import TfidfVectorizer

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline




# Step 1: Load and split text files
"""
csv_files = [r"D:\VS_CODE\INTEL-AIML\Multi_agent_customer_support\support_tickets_10k.csv", r"D:\VS_CODE\INTEL-AIML\Multi_agent_customer_support\faq_knowledge_base_150.csv"]
#csv_files = [r"D:\VS_CODE\INTEL-AIML\Multi_agent_customer_support\support_tickets_10k.csv"]
csv_docs = []

for f in csv_files:
    loader =CSVLoader(f, encoding="utf-8")
    csv_docs.extend(loader.load())

"""



# Example: keep only "ticket_text" and "ticket_category" from support_tickets_10k.csv
df1 = pd.read_csv(
    r"D:\VS_CODE\INTEL-AIML\Multi_agent_customer_support\support_tickets_10k.csv",
    usecols=["ticket_text", "resolution_text"]   
)

# Example: keep only "question" and "answer" from faq_knowledge_base_150.csv
df2 = pd.read_csv(
    r"D:\VS_CODE\INTEL-AIML\Multi_agent_customer_support\faq_knowledge_base_150.csv",
    usecols=["question", "answer"]
)



csv_docs = []

# Convert df1 rows into Documents
for _, row in df1.iterrows():
    csv_docs.append(
        Document(
            page_content=f"{row['ticket_text']} | {row['resolution_text']}",
            metadata={"source": "support_tickets"}
        )
    )

# Convert df2 rows into Documents
for _, row in df2.iterrows():
    csv_docs.append(
        Document(
            page_content=f"Q: {row['question']} A: {row['answer']}",
            metadata={"source": "faq"}
        )
    )




print(type(csv_docs[0]))
#<class 'langchain_core.documents.base.Document'>

print("the length of csv_docs is ",len(csv_docs))
#the length of csv_docs is  10150

print(csv_docs[0])
#It displays the first row as part of page content and the meta data shows the file name and row number


splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_csv_docs = splitter.split_documents(csv_docs)


print("len(split_csv_docs) is ",len(split_csv_docs))
#len(split_csv_docs) is  15792


# Step 5: Convert to embeddings
emb=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",model_kwargs={'device':'cpu'})
vect=FAISS.from_documents(split_csv_docs,emb)
vect.save_local("vector_emb")


#Perform similarity search

#Perform similarity search on FAISS vectorstore.
#Returns a list of Document objects (with page_content + metadata).
    
def vector_search(query, top_val=3):
    
    results = vect.similarity_search(query=query, k=top_val)
    return results



# Initialize Ollama LLM
llm = Ollama(model="llama3")


# --- Chatbot answer (RAG Part) ---
def get_answer(query: str):
    # Step 1: Retrieve relevant docs (assuming vector_search is defined elsewhere)
    docs = vector_search(query, top_val=3)
    context = "\n".join([f"- {d.page_content} | {d.metadata}" for d in docs])

    # Step 2: Build prompt
    prompt = (
        f"""You are a Customer Support assistant. The customer will raise queries related to orders, payments, returns, and delivery
          issues etc.
          Respond in simple sentences.Refer the following examples and answer accordingly.
          
            Question:How do I track my order?
            Answer:Go to My Orders > Select Order > Track Package for real-time courier updates.

            Question:What should I do if my order is delayed?
            Answer:Check tracking first. If no update for 48 hours, raise a complaint via My Orders > Help.
            
            Question:How do I reschedule my delivery?
            Answer:Use the tracking link in your SMS/email to select a new delivery slot.
            
            This is the user query.
            User query: {query}

            Use this relevant context to answer the customer's query.
            Relevant context:{context}"""
        
    )

    try:
        output = llm.invoke(prompt).strip()        
        
    except Exception as e:
        output =  f"Ollama invoke failed: {e}"

    return output






# --- Gradio UI ---
with gr.Blocks() as demo:
    # Intro section with styled text and image
    gr.Markdown(
        """
        <div style="text-align:center; font-size:28px; color:#2E86C1; font-weight:bold;">
            🛒 Multi-Agent Customer Support Intelligence Platform
        </div>
        <div style="text-align:center; font-size:18px; color:#555;">
            Automating ticket classification, routing, response generation, and escalation for E-commerce
        </div>
        """
    )

    gr.Image(
        value="D:\VS_CODE\INTEL-AIML\Multi_agent_customer_support\Copilot_20260907_135808.png",   # replace with your architecture image
        type="filepath",
        label="System Workflow Overview",
        width=400
    )

    gr.Markdown(
        """
        <div style="font-size:16px; color:#1B4F72; text-align:justify;">
            This project demonstrates an <b>AI-powered multi-agent support system</b> built with LangChain, FAISS, and RAG.
            It processes customer tickets for <b>orders, payments, refunds, and delivery issues</b>.
            Agents handle classification, retrieval, response generation, escalation, and continuous learning.
        </div>
        """
    )



# Define mappings
intent_labels = {
    0: "Account & Login",
    1: "App & Website Issue",
    2: "Delivery Issue",
    3: "Payment Issue",
    4: "Product Issue",
    5: "Refund & Return",
    6: "Seller & Product Listing"
}

priority_labels = {
    0: "High",
    1: "Low",
    2: "Medium"
}

# Inside your agent functions
@tool
def intent_agent(text: str) -> str:
    """Classify intent using TF-IDF + Logistic Regression with confidence."""
    intent_model = joblib.load("logreg_intent.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")

    X = vectorizer.transform([text])
    pred_class = intent_model.predict(X)[0]   # integer
    proba = intent_model.predict_proba(X)[0]
    confidence = max(proba)

    label = intent_labels.get(pred_class, str(pred_class))  # convert to text
    return f"{label} (confidence: {confidence:.2f})"



@tool
def priority_agent(text: str) -> str:
    """Classify priority using TF-IDF + Logistic Regression with confidence."""
    priority_model = joblib.load("logreg_priority.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")

    X = vectorizer.transform([text])
    pred_class = priority_model.predict(X)[0]   # integer
    proba = priority_model.predict_proba(X)[0]
    confidence = max(proba)

    label = priority_labels.get(pred_class, str(pred_class))  # convert to text
    return f"{label} (confidence: {confidence:.2f})"


@tool
def sentiment_agent(text: str) -> str:
    """Classify sentiment using fine-tuned DistilBERT (local folder)."""

    # Point to the folder where you saved the model
    model_path = "distilbert_sentiment"

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    # Create sentiment pipeline
    sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

        
    res = sentiment_pipeline(text)[0]
    
    return f"{res['label']} (confidence: {res['score']:.2f})"

   
    

# --- SQL Logging Functions (MySQL) ---
import mysql.connector
from mysql.connector import Error

def create_connection():
    try:
        connection = mysql.connector.connect(
            host="localhost",
            user="root",
            password="root",
            database="multiagent_customer_support"   
        )
        if connection.is_connected():
            print("✅ Connected to MySQL")
            return connection
    except Error as err:
        print(f"Error: {err}")
        return None


def log_to_sql(ticket_text, sentiment, intent, priority, rag_answer):


    print("Logging:", ticket_text, sentiment, intent, priority, rag_answer)

    conn = create_connection()
    if conn:
        cursor = conn.cursor()
        query = """
        INSERT INTO agent_outputs (ticket_text, sentiment, intent, priority, rag_answer)
        VALUES (%s, %s, %s, %s,%s)
                """

              
        cursor.execute(query, (ticket_text, sentiment, intent, priority, rag_answer))
        conn.commit()
        print("✅ Row inserted successfully")
        cursor.close()
        conn.close()





# --- Chat Interface for chatbot ---

import gradio as gr
import time

def agent_pipeline_stream(message, history):
    # Add user message
    history = history + [{"role": "user", "content": message}]

    # Step 1: Sentiment
    history = history + [{"role": "assistant", "content": "🔍 Analyzing sentiment..."}]
    yield history
    time.sleep(1)
    sentiment = sentiment_agent.invoke(message)
    history[-1] = {"role": "assistant", "content": f"✅ Sentiment: {sentiment}"}
    yield history

    # Step 2: Intent
    history = history + [{"role": "assistant", "content": "📊 Classifying intent..."}]
    yield history
    time.sleep(1)
    intent = intent_agent.invoke(message)
    history[-1] = {"role": "assistant", "content": f"✅ Intent: {intent}"}
    yield history

    # Step 3: Priority
    history = history + [{"role": "assistant", "content": "⚡ Determining priority..."}]
    yield history
    time.sleep(1)
    priority = priority_agent.invoke(message)
    history[-1] = {"role": "assistant", "content": f"✅ Priority: {priority}"}
    yield history

    # Step 4: Summary
    summary = f"""
📌 Ticket: {message}
🤖 Agent Decisions
- Sentiment: {sentiment}
- Intent: {intent}
- Priority: {priority}
"""
    history = history + [{"role": "assistant", "content": summary}]
    yield history
    

    # Step 5: RAG Answer
    history = history + [{"role": "assistant", "content": "📚 Retrieving knowledge base suggestions..."}]
    yield history
    time.sleep(1)
    rag_answer = get_answer(message)
    history[-1] = {"role": "assistant", "content": f"📚 RAG Answer\n{rag_answer}"}
    yield history

    #log_to_sql(message, sentiment, intent, priority)
    log_to_sql(message, sentiment, intent, priority, rag_answer)

    
with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox(placeholder="Type your query here...")
    clear = gr.Button("Clear")

    msg.submit(agent_pipeline_stream, [msg, chatbot], [chatbot])
    #clear.click(lambda: [], None, chatbot, queue=False)
    clear.click(lambda: ([], ""), None, [chatbot, msg], queue=False)


demo.launch(inbrowser=True)



