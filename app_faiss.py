import os
import re
import openai
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import PromptLayerChatOpenAI
from langchain.retrievers import EnsembleRetriever
from langchain.vectorstores import FAISS
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from streamlit_player import st_player

# Set OpenAI Model and API key
openai.api_key = st.secrets["OPENAI_API_KEY"]
os.environ["PROMPTLAYER_API_KEY"] = st.secrets["PROMPTLAYER"]
#MODEL = "gpt-3"
#MODEL = "gpt-3.5-turbo"
#MODEL = "gpt-3.5-turbo-0613"
#MODEL = "gpt-3.5-turbo-16k"
MODEL = "gpt-3.5-turbo-16k-0613"
#MODEL = "gpt-4"
#MODEL = "gpt-4-0613"
#MODEL = "gpt-4-32k-0613"

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

st.set_page_config(page_title="Chat with Wardley Videos")
st.title("Chat with Wardley Videos")
st.sidebar.markdown("# Query Videos using AI")
st.sidebar.divider()
st.sidebar.markdown("Developed by Mark Craddock](https://twitter.com/mcraddock)", unsafe_allow_html=True)
st.sidebar.markdown("Current Version: 1.0.0")
st.sidebar.divider()
st.sidebar.markdown("Using GPT-4 API")
st.sidebar.markdown("Uses FAISS")
st.sidebar.markdown("May run out of OpenAI credits")
st.sidebar.divider()
st.sidebar.markdown("Wardley Mapping is provided courtesy of Simon Wardley and licensed Creative Commons Attribution Share-Alike.")

# Get datastore
YT_DATASTORE = "datastore/simon"
BOOK_DATASTORE = "datastore/book"

if os.path.exists(YT_DATASTORE):
    yt_index = FAISS.load_local(
        f"{YT_DATASTORE}",
        OpenAIEmbeddings()
    )
else:
    st.write(f"Missing files. Upload index.faiss and index.pkl files to {YT_DATASTORE} directory first")

if os.path.exists(BOOK_DATASTORE):
    yt_index = FAISS.load_local(
        f"{BOOK_DATASTORE}",
        OpenAIEmbeddings()
    )
else:
    st.write(f"Missing files. Upload index.faiss and index.pkl files to {BOOK_DATASTORE} directory first")

yt_retriever = yt_index.as_retriever(search_type="mmr", search_kwargs={"k": 5})
book_retriever = book_index.as_retriever(search_type="mmr", search_kwargs={"k": 5})

# initialize the ensemble retriever
ensemble_retriever = EnsembleRetriever(retrievers=[yt_retriever, books_retriever], weights=[0.5, 0.5])

system_template="""
    As a chatbot, analyze the provided videos on Wardley Mapping and offer insights and recommendations.
    Suggestions:
    Explain the analysis process for a Wardley Map
    Discuss the key insights derived from the videos
    Provide recommendations based on the analysis
    Use the following pieces of context to answer the users question.
    If you don't know the answer, just say that "I don't know", don't try to make up an answer.
    ----------------
    {summaries}"""
prompt_messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}")
    ]
prompt = ChatPromptTemplate.from_messages(prompt_messages)

chain_type_kwargs = {"prompt": prompt}
llm = PromptLayerChatOpenAI(
    model_name=MODEL,
    temperature=0,
    max_tokens=2000,
    pl_tags=["wardleygptv2"],
)  # Modify model_name if you have access to GPT-4

chain = RetrievalQAWithSourcesChain.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 3}), # Use MMR search and return 5 (max 20) video sources
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs
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
            #response = chain(query)
            response = ensemble_retriever.get_relevant_documents("what is inertia")
            st.markdown(response['answer'])
            st.divider()
            
            for index, document in enumerate(response):
                if 'source' in document.metadata:
                    metadata = document.metadata
                    st.write(f"\nSource {index + 1}: {metadata.get('source', 'Unknown')}")
                    st.write(f"Video title: {metadata.get('title', 'Unknown')}")
                    st.write(f"Video author: {metadata.get('author', 'Unknown')}")
                    start_time = int(metadata.get('start_time', 0))
                    st.write(f"Source video: https://youtu.be/{metadata.get('source_video', 'Unknown')}?t={start_time}")
                    st.write(f"Start Time: {metadata.get('start_time', '0')}")
                    cleaned_content = clean_text(document.page_content)
                    st.write(f"Content: {cleaned_content}\n")
                    video_id = f"Source video: https://youtu.be/{metadata.get('source_video', 'Unknown')}?t={start_time}"
                    key = f"video_{index}"
                    st_player(video_id, height=150, key=key)

        st.session_state.messages.append({"role": "assistant", "content": response['answer']})