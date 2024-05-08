
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
import os
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
import streamlit as st
import time
from langchain_together import Together

st.set_page_config(page_title="GymGPT")
col1, col2, col3 = st.columns([1,2,1])
with col2:
    st.title("GymGPT 🦾")


st.sidebar.title("Welcome to GymGPT")
st.sidebar.image("WE_GO_JIM.png", width=275)
st.sidebar.title("Shoot your gym-related questions")
st.markdown(
    """
    <style>
div.stButton > button:first-child {
    background-color: #ffd0d0;
}

div.stButton > button:active {
    background-color: #ff6262;
}

.st-emotion-cache-6qob1r {
    position: relative;
    height: 100%;
    width: 100%;
    background-color: black;
    overflow: overlay;
}

   div[data-testid="stStatusWidget"] div button {
        display: none;
        }
    
    .reportview-container {
            margin-top: -2em;
        }
        #MainMenu {visibility: hidden;}
        .stDeployButton {display:none;}
        footer {visibility: hidden;}
        #stDecoration {display:none;}
    button[title="View fullscreen"]{
    visibility: hidden;}
        </style>
""",
    unsafe_allow_html=True,
)

def reset_conversation():
    st.session_state.messages = []
    st.session_state.memory.clear()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(k=2, memory_key="chat_history",return_messages=True) 

embeddings = HuggingFaceEmbeddings(model_name="nomic-ai/nomic-embed-text-v1-ablated", model_kwargs={"trust_remote_code": True})
db = FAISS.load_local("gym_vector_db", embeddings)
db_retriever = db.as_retriever(search_type="similarity",search_kwargs={"k": 4})

prompt_template = """<s>[INST]This is a chat template and you are the gym trainer, your primary objective is to provide accurate and concise information related to gym, workout, bodybuilding based on the user's questions. Do not generate your own questions and answers. You will adhere strictly to the instructions provided, offering relevant context from the knowledge base while avoiding unnecessary details. Your responses will be brief, to the point, and in compliance with the established format. If a question falls outside the given context, rely on your own knowledge base to generate an appropriate response. You will prioritize the user's query and refrain from posing additional questions and do not repeat the prompt template and the things that you have said already.
QUESTION: {question}
CONTEXT: {context}
CHAT HISTORY: {chat_history}[/INST]
ASSISTANT:
</s>
"""

prompt = PromptTemplate(template=prompt_template,
                        input_variables=['question', 'context', 'chat_history'])

llm = Together(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    temperature=0.7,
    max_tokens=1024,
    top_k=1,
    together_api_key=os.environ['T_API']
)

qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    memory=st.session_state.memory,
    retriever=db_retriever,
    combine_docs_chain_kwargs={'prompt': prompt}
)

for message in st.session_state.messages:
    with st.chat_message(message.get("role")):
        st.write(message.get("content"))

input_prompt = st.chat_input("Say something")

if input_prompt:
    with st.chat_message("user"):
        st.write(input_prompt)

    st.session_state.messages.append({"role":"user","content":input_prompt})

    with st.chat_message("assistant"):
        with st.status("Lifting data, one bit at a time 💡🦾...",expanded=True):
            result = qa.invoke(input=input_prompt)

            message_placeholder = st.empty()

            full_response = "⚠️ **_Note: Information provided may be inaccurate._** \n\n\n"
        for chunk in result["answer"]:
            full_response+=chunk
            time.sleep(0.02)
            
            message_placeholder.markdown(full_response+" ▌")
        st.button('Reset All Chat 🗑️', on_click=reset_conversation)

    st.session_state.messages.append({"role":"assistant","content":result["answer"]})
