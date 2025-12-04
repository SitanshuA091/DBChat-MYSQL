import streamlit as st
import getpass
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.utilities import SQLDatabase
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from PIL import Image
import mysql.connector

from langchain_groq import ChatGroq

load_dotenv()

os.getenv("GROQ_API_KEY")

#if "GROQ_API_KEY" not in os.environ:
#    os.environ["GROQ_API_KEY"] = getpass.getpass("Enter your Groq API key: ")
    
#os.environ["LANGSMITH_API_KEY"] = getpass.getpass("Enter your LangSmith API key: ")
#os.environ["LANGSMITH_TRACING"] = "true"

Icon_img = Image.open("icon.png")

st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Merriweather:ital,wght@0,300;0,400;0,700;0,900;1,300;1,400;1,700;1,900&family=Space+Mono:ital,wght@0,400;0,700;1,400;1,700&display=swaphttps://fonts.googleapis.com/css2?family=Merriweather:ital,wght@0,300;0,400;0,700;0,900;1,300;1,400;1,700;1,900&family=Space+Mono:ital,wght@0,400;0,700;1,400;1,700&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)
LOGO_URL = "https://registry.npmmirror.com/@lobehub/icons-static-png/1.65.0/files/dark/grok.png"
st.logo(LOGO_URL, size="large", link="https://registry.npmmirror.com/@lobehub/icons-static-png/1.65.0/files/dark/grok.png")

st.set_page_config(page_title = "DB-Chat", 
                   page_icon=Icon_img,
                   layout="wide",
    initial_sidebar_state="expanded")

st.title("Chat with MYSQL Database")

def init_database(user: str, password: str, host: str, port: str, database: str) -> SQLDatabase:
  db_uri = f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}"
  return SQLDatabase.from_uri(db_uri)

def get_sql_chain(db):
  template = """
    You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
    Based on the table schema below, write a SQL query that would answer the user's question. Take the conversation history into account.
    
    <SCHEMA>{schema}</SCHEMA>
    
    Conversation History: {chat_history}
    
    Write only the SQL query and nothing else. Do not wrap the SQL query in any other text, not even backticks.
    
    For example:
    Question: which 3 artists have the most tracks?
    SQL Query: SELECT ArtistId, COUNT(*) as track_count FROM Track GROUP BY ArtistId ORDER BY track_count DESC LIMIT 3;
    Question: Name 10 artists
    SQL Query: SELECT Name FROM Artist LIMIT 10;
    
    Your turn:
    
    Question: {question}
    SQL Query:
    """
    
  prompt = ChatPromptTemplate.from_template(template)
  
  model = ChatGroq( model="openai/gpt-oss-120b",
    temperature=0)
  
  def get_schema(_):
    return db.get_table_info()
  
  return (
    RunnablePassthrough.assign(schema=get_schema)
    | prompt
    | model
    | StrOutputParser()
  )
    
def get_response(user_query: str, db: SQLDatabase, chat_history: list):
  sql_chain = get_sql_chain(db)
  
  template = """
    You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
    Based on the table schema below, question, sql query, and sql response, write a natural language response.
    <SCHEMA>{schema}</SCHEMA>

    Conversation History: {chat_history}
    SQL Query: <SQL>{query}</SQL>
    User question: {question}
    SQL Response: {response}"""
  
  prompt = ChatPromptTemplate.from_template(template)
  
  model = ChatGroq(model="openai/gpt-oss-120b",
    temperature=0)
  
  chain = (
    RunnablePassthrough.assign(query=sql_chain).assign(
      schema=lambda _: db.get_table_info(),
      response=lambda vars: db.run(vars["query"]),
    )
    | prompt
    | model
    | StrOutputParser()
  )
  
  return chain.invoke({
    "question": user_query,
    "chat_history": chat_history,
  })
    
  
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
      AIMessage(content="Hello! I'm a SQL assistant. Ask me anything about your database."),
    ]
with st.sidebar:
    st.subheader("Settings")
    st.write("This is a simple chat application using MySQL. Connect to the database and start chatting.")
    
    st.text_input("Host", value="localhost", key="Host")
    st.text_input("Port", value="3306", key="Port")
    st.text_input("User", value="root", key="User")
    st.text_input("Password", type="password", value="admin", key="Password")
    st.text_input("Database", key="Database")
    
    if st.button("Connect"):
        with st.spinner("Connecting to database"):
            db = init_database(
                st.session_state["User"],
                st.session_state["Password"],
                st.session_state["Host"],
                st.session_state["Port"],
                st.session_state["Database"]
            )
            st.session_state.db = db
            st.success("✅ Database connection successfully")
            
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI", avatar="https://img.icons8.com/?size=100&id=8SQ5YC8kxJOf&format=png&color=FFFFFF"):
            st.markdown(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)

user_query = st.chat_input("Type a message...")
if user_query is not None and user_query.strip() != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    
    with st.chat_message("Human"):
        st.markdown(user_query)
        
    with st.chat_message("AI",  avatar="https://img.icons8.com/?size=100&id=8SQ5YC8kxJOf&format=png&color=FFFFFF"):
        response = get_response(user_query, st.session_state.db, st.session_state.chat_history)
        st.markdown(response)
        
    st.session_state.chat_history.append(AIMessage(content=response))