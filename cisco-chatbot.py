import sys

import streamlit as st

from openai import AzureOpenAI

from utils.es_helper import create_es_client
from utils.openai_helper import get_chat_guidance
from utils.query_helper import search_products_for_chatbot, search_products_v2
from variables import openai_api_version, openai_api_sa_base

# Initialize these variables with default values at the start of the script
BM25_Boost = 0
KNN_Boost = 0
rrf_rank_constant = 1
rrf_window_size = 200

first_response_text = None

# Connect to Elasticsearch
try:
    username = st.secrets['es_username']
    password = st.secrets['es_password']
    cloudid = st.secrets['es_cloudid']
    es = create_es_client(username, password, cloudid)
except Exception as e:
    print("Connection failed", str(e))
    st.error("Error connecting to Elasticsearch. Fix connection and restart app")
    sys.exit(1)


azureclient = AzureOpenAI(
  api_key = st.secrets['sa_pass'],
  api_version = openai_api_version,
  azure_endpoint = openai_api_sa_base
)

searchtype = 'Elser'

st.markdown("""
        <style>
        .subheader-style {
            font-size: 40px;
            font-weight: bold;
            color: #0489ba;  /* Change to your preferred color */
            text-shadow: 2px 2px 5px #7F7F7F;  /* Adding a subtle shadow for depth */
            margin-bottom: 25px;  /* Optional: Adjusts space below the subheader */
        }
        </style>
        <div class='subheader-style'>Cisco ELSER Infused Chat Bot</div>
        """, unsafe_allow_html=True)





if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "You are an AI assistant. Your answers should stay short and concise. Explain your answer. No formalities. Do no use any information outside of the content I provide"}
    ]

for message in st.session_state.messages:
    # Skip messages that contain specific substrings
    if "You are an AI assistant" in message["content"] or "find the answer using this content only" in message["content"]:
        continue
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("How may I help you?"):
    #st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)


    blog_bodies= search_products_v2(es, prompt, searchtype, rrf_rank_constant, rrf_window_size)


    with st.spinner('Thinking...'):
        # Initially, set response_text to None or an appropriate default value
        response_text = None

        print(len(st.session_state.messages))

        # Check if the session state has more than just the initial system message
        # Assuming the system message is the first item in the list
        if len(st.session_state.messages) == 1:
            print("ini retrival")
            st.session_state.messages.append({"role": "user", "content": f"find the answer using this content only:  {blog_bodies}"})
            st.session_state.messages.append({"role": "user", "content": f"{prompt}"})
            response_text = get_chat_guidance(azureclient)
        else:
            print("no retrival")
            st.session_state.messages.append({"role": "user", "content": f"{prompt}"})
            response_text = get_chat_guidance(azureclient)

    # Display the assistant's response in the chat
    with st.chat_message("assistant"):
        st.markdown(response_text)

    # Append the assistant's response to the session state
    st.session_state.messages.append({"role": "assistant", "content": response_text})
