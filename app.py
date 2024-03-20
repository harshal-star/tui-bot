import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader
from langchain.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
import streamlit as st
import pinecone
from langchain.vectorstores import Pinecone

st.set_page_config(layout = "wide")

# Sidebar for API key input
api_key = st.sidebar.text_input("Enter your OpenAI API key:", type="password", key="api_key_input")
pinecone_api_key = st.sidebar.text_input("Enter your Pinecone API key:", type="password", key="pinecone_api_key_input")
pinecone_env = st.sidebar.text_input("Enter your Pinecone environment name:", type="password", key="pinecone_env_key_input")
index_name = st.sidebar.text_input("Enter your Pinecone Index name:", type="password", key="piencone_index_key_input")
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key

if pinecone_api_key:
    PINECONE_API_KEY = pinecone_api_key

if pinecone_env:
    PINECONE_ENV = pinecone_env

if index_name:
    index_name = index_name


with st.sidebar:
   st.header('About the App')
   st.write("""Welcome to the TUI Interactive FAQ Chatbot, powered by Streamlit. This tool is your digital guide to all things TUI, offering 
      instant answers from our extensive FAQ database. Whether you have questions about booking, destinations, policies, or services, our chatbot leverages 
      advanced natural language processing to provide precise, relevant information. Simplify your travel planning and get support in real-time with our 
      user-friendly chatbot, designed to enhance your TUI experience at every step.""")

data_path = r"data/"

files = os.listdir(data_path)


agent_template = """As an agent, your role is to provide clear, informed, and helpful responses to frequently asked questions. Draw on the provided 
context to offer solutions or guidance, ensuring your answers reflect your expertise and authority in the subject matter. If the information is 
unavailable, offer the next best steps or admit the limitation with professionalism. The responses should be concise, aiming for no more than three 
sentences.
Question: {question}
Context: {context}
Answer:
"""

customer_template = """As a customer, your inquiries or responses reflect common questions, concerns, or feedback based on your experiences or the 
need for information. Use the provided context to express your perspective, highlighting any confusion, satisfaction, or desire for further clarification 
in a personal and relatable manner. If the context does not fully address your concerns, frame your response to seek additional help or information. 
Keep your answers to three sentences maximum, ensuring they are direct and reflective of a customer’s viewpoint.
Question: {question}
Context: {context}
Answer:
"""

def document_data(query, chat_history):
   # all_documents = []
   # for file in files:
   #     file_path = os.path.join(data_path, file)
   #     loader = TextLoader(file_path)
   #     documents = loader.load()
   #     all_documents.append(documents)

   # final_document = [item for sublist in all_documents for item in sublist]
   # text_splitter = RecursiveCharacterTextSplitter(chunk_size = 100, chunk_overlap = 50, separators = ["\n\n","\n"," ",""]) 
   # text = text_splitter.split_documents(documents = final_document) 

   # creating embeddings using OPENAI

   embeddings = OpenAIEmbeddings()

   # Initializing Pinecone with the correct API key and environment
   pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

   # vectorstore = FAISS.from_documents(text, embeddings)
   # vectorstore.save_local("vectors")
   # print("Embeddings successfully saved in vector Database and saved locally")

   # # Loading the saved embeddings 
   # loaded_vectors=FAISS.load_local("vectors", embeddings)

   # Create vector store using Pinecone
   # vectorstore = Pinecone.from_documents(text, embeddings, index_name = index_name, namespace = "tui")
   vectorstore = Pinecone.from_existing_index(index_name, embeddings)

   # ConversationalRetrievalChain 
   qa = ConversationalRetrievalChain.from_llm(
       llm = ChatOpenAI(temperature=0.1, model_name="gpt-4-1106-preview"), 
       retriever =  vectorstore.as_retriever(search_kwargs={'k': 6}, namespace = "tui"),
       combine_docs_chain_kwargs={"prompt": custom_prompt}
    )
    
   return qa({"question":query, "chat_history":chat_history})

if __name__ == '__main__':

   st.header("FAQs chatbot")

   # Add this dropdown to choose between agent and customer personas
   persona = st.sidebar.selectbox(
        'Choose your persona',
        ('Agent', 'Customer')
    )

   # Then, based on the selected persona, choose the template
   if persona == 'Agent':
      template = agent_template
   else:
      template = customer_template

   # Modify your custom_prompt instance to use the selected template
   custom_prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"],
    )
    # ChatInput
   prompt = st.chat_input("Enter your questions here")

   if "user_prompt_history" not in st.session_state:
      st.session_state["user_prompt_history"]=[]
   if "chat_answers_history" not in st.session_state:
      st.session_state["chat_answers_history"]=[]
   if "chat_history" not in st.session_state:
      st.session_state["chat_history"]=[]

   if prompt:
      with st.spinner("Generating......"):
         output=document_data(query=prompt, chat_history = st.session_state["chat_history"])

         # Storing the questions, answers and chat history

         st.session_state["chat_answers_history"].append(output['answer'])
         st.session_state["user_prompt_history"].append(prompt)
         st.session_state["chat_history"].append((prompt,output['answer']))

   # Displaying the chat history

   if st.session_state["chat_answers_history"]:
      for i, j in zip(st.session_state["chat_answers_history"],st.session_state["user_prompt_history"]):
         message1 = st.chat_message("user")
         message1.write(j)
         message2 = st.chat_message("assistant")
         message2.write(i)