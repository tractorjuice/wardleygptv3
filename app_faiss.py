import os
import re
import uuid
from langchain_openai import OpenAI
import streamlit as st
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from streamlit_player import st_player

MODEL = "gpt-3.5-turbo"
DEBUG = True # True to overwrite files that already exist

# Remove HTML from sources
def remove_html_tags(text):
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

def remove_markdown(text):
    # Remove headers (e.g., # Header)
    text = re.sub(r'#.*$', '', text, flags=re.MULTILINE)
    # Remove bold/italic (e.g., **bold**, *italic*)
    text = re.sub(r'\*.*\*', '', text)
    # Remove links (e.g., [text](url))
    text = re.sub(r'\[.*\]\(.*\)', '', text)
    # Remove lists (e.g., - item)
    text = re.sub(r'- .*$', '', text, flags=re.MULTILINE)
    return text

def clean_text(text):
    text = remove_html_tags(text)
    text = remove_markdown(text)
    return text

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []
    
st.set_page_config(page_title="Chat with Wardley")
st.title("Chat with Wardley")
st.sidebar.markdown("# Query YouTube Videos & Books using AI")
st.sidebar.divider()
st.sidebar.markdown("Developed by Mark Craddock](https://twitter.com/mcraddock)", unsafe_allow_html=True)
st.sidebar.markdown("Current Version: 1.3.1")
st.sidebar.markdown("Wardley Mapping is provided courtesy of Simon Wardley and licensed Creative Commons Attribution Share-Alike.")
st.sidebar.markdown(st.session_state.session_id)
st.sidebar.divider()

# Check if the user has provided an API key, otherwise default to the secret
user_openai_api_key = st.sidebar.text_input("Enter your OpenAI API Key:", placeholder="sk-...", type="password")

# Get datastore
YT_DATASTORE = "datastore/simon"
BOOK_DATASTORE = "datastore/book"

if user_openai_api_key:
    os.environ["OPENAI_API_KEY"] = user_openai_api_key
    
    if "yt_index" not in st.session_state:
        if os.path.exists(YT_DATASTORE):
            st.session_state.yt_index = FAISS.load_local(
                YT_DATASTORE,
                OpenAIEmbeddings(),
                allow_dangerous_deserialization='True',
            )
        else:
            st.write(f"Missing files. Upload index.faiss and index.pkl files to {DATA_STORE_DIR} directory first")
    
        if os.path.exists(BOOK_DATASTORE):
            st.session_state.book_index = FAISS.load_local(
                BOOK_DATASTORE,
                OpenAIEmbeddings(),
                allow_dangerous_deserialization='True',
            )
        else:
            st.write(f"Missing files. Upload index.faiss and index.pkl files to {DATA_STORE_DIR} directory first")
    
    
        custom_system_template="""
            As a friendly and helpful assistant with expert knowledge in Wardley Mapping,
            Analyze the provided book on Wardley Mapping and offer insights and recommendations.
            Suggestions:
            Explain the analysis process for a Wardley Map
            Discuss the key insights derived from the book
            Provide recommendations based on the analysis
            Use the following pieces of context to answer the users question.
            If you don't know the answer, just say that "I don't know", don't try to make up an answer.
            Your primary objective is to help the user formulate excellent answers by utilizing the context about the book and 
            relevant details from your knowledge, along with insights from previous conversations.
            ----------------
            Reference Context and Knowledge from Similar Existing Services: {context}
            Previous Conversations: {chat_history}"""
        
        custom_user_template = "Question:'''{question}'''"
        
        prompt_messages = [
            SystemMessagePromptTemplate.from_template(custom_system_template),
            HumanMessagePromptTemplate.from_template(custom_user_template)
            ]
        prompt = ChatPromptTemplate.from_messages(prompt_messages)
           
        yt_retriever = st.session_state.yt_index.as_retriever(search_type="mmr", search_kwargs={"k": 2})
        book_retriever = st.session_state.book_index.as_retriever(search_type="mmr", search_kwargs={"k": 2})
        # initialize the ensemble retriever
        st.session_state.ensemble_retriever = EnsembleRetriever(retrievers=[yt_retriever, book_retriever], weights=[0.5, 0.5])
    
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True, output_key='answer')
    
    if "llm" not in st.session_state:
        st.session_state.llm = ChatOpenAI(
            model_name=MODEL,
            temperature=0,
            max_tokens=400,
        )
    
    if "chain" not in st.session_state:
    
        st.session_state.chain = ConversationalRetrievalChain.from_llm(
            llm=st.session_state.llm,
            retriever=st.session_state.ensemble_retriever,
            chain_type="stuff",
            rephrase_question = True,
            return_source_documents=True,
            memory=st.session_state.memory,
            combine_docs_chain_kwargs={'prompt': prompt}
        )
        
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        if message["role"] in ["user", "assistant"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    if query := st.chat_input("What question do you have for the videos?"):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
    
        with st.spinner():
            with st.chat_message("assistant"):
                response = st.session_state.ensemble_retriever.get_relevant_documents(query)
                #response = st.session_state.chain(query)
                #st.markdown(response['answer'])
                st.divider()
                
                for index, document in enumerate(response):
                    if 'source' in document.metadata:
                        metadata = document.metadata
                        source = metadata.get('source', 'Unknown')
                        
                        # Check if the source string contains the given path
                        if '/AI/WardleyKB/books/book/Wardley Maps' in source:
                            source = 'Simon Wardley Book'
    
                        cleaned_content = clean_text(document.page_content)
                        st.write(f"Content: {cleaned_content}\n")
    
                        if source == 'YouTube':
                            start_time = int(metadata.get('start_time', 0))
                            video_id = f"Source video: https://youtu.be/{metadata.get('source_video', 'Unknown')}?t={start_time}"
                            key = f"video_{index}"
                            st_player(video_id, height=150, key=key)
                            with st.expander(f"\nSource {metadata.get('title', 'Unknown')}"):
                                st.write(f"\nSource {index + 1}: {source}")
                                st.write(f"Video title: {metadata.get('title', 'Unknown')}")
                                st.write(f"Video author: {metadata.get('author', 'Unknown')}")
                                st.write(f"Source video: https://youtu.be/{metadata.get('source_video', 'Unknown')}?t={start_time}")
                                st.write(f"Start Time: {metadata.get('start_time', '0')}")
    
                        if source == 'Simon Wardley Book':
                            with st.expander(f"\nSource {index + 1}: {source}"):
                                st.write(f"\nSource {index + 1}: {source}")
                                st.write(f"Page: {metadata.get('page', 'Unknown')}")
                            
                        st.divider()
