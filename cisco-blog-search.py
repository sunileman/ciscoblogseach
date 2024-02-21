# Import the required libraries
import math
import sys
from utils.es_helper import create_es_client
import streamlit as st

from utils.query_helper import search_products

# Initialize these variables with default values at the start of the script
BM25_Boost = 0
KNN_Boost = 0
rrf_rank_constant = 1
rrf_window_size = 200

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

# Layout columns
col1, col2 = st.columns([1, 3])  # col1 is 1/4 of the width, and col2 is 3/4


# Initialize the search clicked state
if 'search_clicked' not in st.session_state:
    st.session_state.search_clicked = False

with col2:
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
        <div class='subheader-style'>Cisco Blog Search</div>
        """, unsafe_allow_html=True)


    # Sub-columns within col2
    col2a, col2b = st.columns([3, 1.1])

    with col2a:
        # Search type radio
        searchtypeselected = col2a.radio(
            "Search Method",
            ("Keyword", "Semantic", "Hybrid (Keyword & Semantic) + AutoRank"),
            index=0  # Default to "BM25"
        )

        if searchtypeselected == "Keyword":
            searchtype = 'BM25'
        elif searchtypeselected == "Semantic":
            searchtype = 'Semantic'
        elif searchtypeselected == "Hybrid (Keyword & Semantic)":
            searchtype = 'Vector OpenAI Hybrid'
        elif searchtypeselected == "Hybrid (Keyword & Semantic) + AutoRank":
            searchtype = 'Reciprocal Rank Fusion'

        user_query = col2a.text_area("Product Search")

        # ... [rest of your code for searching, displaying results, etc.]

    with col2b:
        # Conditionally display sliders based on `searchtype` value
        BM25_Boost = 0
        KNN_Boost = 0
        if searchtype == "Semantic":
            embedding_selection = col2b.radio(
                "Semantic Models:",
                ("ELSER", "MiniLM-L6"),
                index=0  # MiniLM-L6
            )

            if embedding_selection == "MiniLM-L6":
                searchtype = 'Vector'
            elif embedding_selection == "ELSER":
                searchtype = 'Elser'
            else:
                searchtype = 'Vector OpenAI'

        if searchtype == "Vector Hybrid" or searchtype == "Elser Hybrid" or searchtype == "Vector OpenAI Hybrid":
            BM25_Boost = col2b.slider(
                "BM25 Score Boost:",
                min_value=0.0,
                max_value=5.0,
                value=0.00,
                step=0.01,
                help="Adjust the BM25 score boost value. Higher values give more weight to BM25 scores."
            )

            KNN_Boost = col2b.slider(
                "KNN Score Boost:",
                min_value=0.0,
                max_value=5.0,
                value=0.00,
                step=0.01,
                help="Adjust the KNN score boost value. Higher values give more weight to KNN scores."
            )



    # Check if searchtype has changed
    if searchtype != st.session_state.get('previous_searchtype', None):
        st.session_state.search_clicked = False
        st.session_state.previous_searchtype = searchtype

    # user_query = st.text_area("Tariff Search")

    if st.button("Search"):
        if user_query:


            processed_results, original_results = search_products(es, user_query, searchtype, BM25_Boost, KNN_Boost,
                                                                  rrf_rank_constant, rrf_window_size)
            if searchtype != "GenAI":

                first_instance = processed_results[0]

                # Check if there are any results
                if not processed_results:  # This checks if the list is empty
                    st.info("No search results found.")


                query_response_time_first_instance = first_instance[3]
                total_genai_query_time = math.ceil(sum(result[3] for result in processed_results))


                st.markdown(
                    f'<div style="color: lightgreen">Elastic Query Response Time: {query_response_time_first_instance}ms   |    GenAI Time: {total_genai_query_time}ms</div><br><br>',
                    unsafe_allow_html=True)

                result_counter = 1  # Initialize the counter
                for idx, (text, completion_output, score, query_response_time, ga_unused, url, title, first_passage_text) in enumerate(
                        processed_results):
                    # Check if 'url' exists and create a hyperlink
                    if url:
                        hyperlink = f'<a href="{url}" target="_blank">{title}</a>'  # Create hyperlink to open in a new tab
                        st.markdown(
                            f'<center>{hyperlink}</center><br>',
                            unsafe_allow_html=True)

                    badge_style = "background-color: red; color: white; border-radius: 50%; padding: 5px 10px;"
                    st.markdown(
                        f'<span style="{badge_style}">{result_counter}</span> AI Insight/Summary: {completion_output}',
                        unsafe_allow_html=True)
                    score_str = f"{score:.2f}" if score else "Not Applicable"
                    st.markdown("**Supporting Document Excerpt:**")
                    # Use markdown to create a scrollable text area for 'text'
                    st.markdown(
                        f'<div style="height:100px;overflow-y:scroll;padding:10px;border:1px solid gray;">{first_passage_text}</div>',
                        unsafe_allow_html=True)
                    st.markdown('<hr style="border-top: 3px solid white">', unsafe_allow_html=True)
                    # Increment the counter for the next result
                    result_counter += 1
            else:
                for idx, (completion_output) in enumerate(
                        processed_results):
                    st.markdown(f"**AI Insight:** {completion_output}")

        else:
            st.error("Please enter a question before searching.")
