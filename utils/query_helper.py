import json

from variables import  model, vector_embedding_field, byom_index_name


from utils.openai_helper import get_chat_guidance_rag, get_openai_large_guidance


def build_bm25_query(user_query):
    # Constructing the match query for 'organic' search using 'body_content'
    bm25_query = {
        "bool": {
            "should": [
                {
                    "query_string": {
                        "default_field": "body_content",
                        "query": user_query
                    }
                }
            ]
        }
    }

    full_query = {
        "size": 5,  # Specify the number of results to return
        "query": bm25_query
    }

    # Debug: Dump the assembled query for inspection
    print(json.dumps(full_query, indent=4))

    return full_query


def build_openai_hybrid_query(embeddings, user_query, BM25_Boost, KNN_Boost):
    """
    Builds a hybrid Elasticsearch query based on the provided parameters.

    Returns:
    - A dictionary representing the Elasticsearch query.
    """

    knn_query = {
        "field": vector_embedding_field,  # Field containing the OpenAI embeddings
        "k": 10,
        "num_candidates": 100,
        "query_vector": embeddings,
        "boost": KNN_Boost
    }

    main_filters = []

    # Copy the main_filters for the knn part
    knn_filters = list(main_filters)

    query = {
        "query": {
            "bool": {
                "must": {
                    "match": {
                        "combined_relevancy": {
                            "query": user_query,
                            "boost": BM25_Boost
                        }
                    }
                },
                "filter": main_filters
            }
        },
        "knn": knn_query
    }

    if knn_filters:
        query["knn"]["filter"] = knn_filters

    print(json.dumps(query, indent=4))

    return query



def build_vector(es, text):
    docs = [{"text_field": text}]
    response = es.ml.infer_trained_model(model_id=model, docs=docs)

    predicted_value = response.get('inference_results', [{}])[0].get('predicted_value', [])

    print(predicted_value)
    return predicted_value


def build_knn_query(user_query, query_vector):
    """
    Builds an updated Elasticsearch KNN query for nested structures with query vectors.

    Parameters:
    - user_query: The query text input by the user. (Not used in this specific function but kept for compatibility)
    - query_vector: The precomputed vector for the KNN query.

    Returns:
    - A dictionary representing the Elasticsearch KNN nested query.
    """

    # Nested KNN query structure
    nested_knn_query = {
        "query": {
            "nested": {
                "path": "passages",
                "query": {
                    "knn": {
                        "query_vector": query_vector,
                        "field": "passages.vector.predicted_value",
                        "num_candidates": 2
                    }
                },
                "inner_hits": {
                    "_source": [
                        "passages.text"
                    ]
                }
            }
        }
    }

    # Debug: Dump the assembled query for inspection
    print(json.dumps(nested_knn_query, indent=4))

    return nested_knn_query


def build_rrf_query(embeddings, user_query, rrf_rank_constant, rrf_window_size):
    """
    Builds a complex query with sub_searches including match, knn, and text_expansion queries,
    and aggregates results using RRF.

    Parameters:
    - embeddings: The precomputed vector for the KNN query.
    - user_query: The query text input by the user.
    - rrf_rank_constant: The rank constant used in RRF ranking.
    - rrf_window_size: The window size used in RRF ranking.

    Returns:
    - A dictionary representing the complex query.
    """

    # Define the base structure of the query
    query = {
        "sub_searches": [
            {
                "query": {
                    "match": {
                        "body_content": user_query
                    }
                }
            },
            {
                "query": {
                    "nested": {
                        "path": "passages",
                        "query": {
                            "knn": {
                                "query_vector": embeddings,
                                "field": "passages.vector.predicted_value",
                                "num_candidates": 50
                            }
                        }
                    }
                }
            },
            {
                "query": {
                    "nested": {
                        "path": "passages",
                        "query": {
                            "bool": {
                                "should": [
                                    {
                                        "text_expansion": {
                                            "passages.content_embedding.predicted_value": {
                                                "model_id": ".elser_model_2_linux-x86_64",
                                                "model_text": user_query
                                            }
                                        }
                                    }
                                ]
                            }
                        }
                    }
                }
            }
        ],
        "rank": {
            "rrf": {
                "window_size": rrf_window_size,
                "rank_constant": rrf_rank_constant
            }
        }
    }
    # Debug: Print the assembled query for inspection
    print(json.dumps(query, indent=4))

    return query


def build_elser_query(user_query):
    # Nested query with text_expansion
    nested_query = {
        "nested": {
            "path": "passages",
            "query": {
                "bool": {
                    "should": [
                        {
                            "text_expansion": {
                                "passages.content_embedding.predicted_value": {
                                    "model_id": ".elser_model_2_linux-x86_64",
                                    "model_text": user_query
                                }
                            }
                        }
                    ]
                }
            }
        }
    }

    query = {
        "size": 5,
        "query": nested_query
    }

    # Debug: Print the assembled query for inspection
    print(json.dumps(query, indent=4))

    return query




def search_products(es, user_query, searchtype, BM25_Boost, KNN_Boost, rrf_rank_constant, rrf_window_size):
    # Select the appropriate query building function based on searchtype
    if searchtype == "Vector":
        query = build_knn_query(user_query, build_vector(es, user_query))
    elif searchtype == "BM25":
        query = build_bm25_query(user_query)
    elif searchtype == "Reciprocal Rank Fusion":
        query = build_rrf_query(build_vector(es, user_query), user_query, rrf_rank_constant, rrf_window_size)
    elif searchtype == "Elser":
        query = build_elser_query(user_query)
    else:
        raise ValueError(f"Invalid searchtype: {searchtype}")

    results = es.search(index=byom_index_name, body=query, _source=True)

    # Set a default value for num_results
    num_results = 0

    if results and results.get('hits') and results['hits'].get('hits'):
        # Limit the number of results displayed

        num_results = min(5, len(results['hits']['hits']))

        for i in range(num_results):
            if results['hits']['hits'][i]["_source"].get("passages") and len(
                    results['hits']['hits'][i]["_source"]["passages"]) > 0:
                first_passage_text = results['hits']['hits'][i]["_source"]["passages"][0]["text"]
            else:
                first_passage_text = "No passages text available"

            print(first_passage_text)

    else:
        print("No results found.")

    if searchtype != "GenAI":
        return get_openai_large_guidance(user_query, results, num_results, searchtype)


def search_products_for_chatbot(es, user_query, searchtype, rrf_rank_constant, rrf_window_size,
                                azureclient, conversation_history):
    # Select the appropriate query building function based on searchtype
    if searchtype == "Vector":
        query = build_knn_query(user_query, build_vector(es, user_query))
    elif searchtype == "BM25":
        query = build_bm25_query(user_query)
    elif searchtype == "Reciprocal Rank Fusion":
        query = build_rrf_query(build_vector(es, user_query), user_query, rrf_rank_constant, rrf_window_size)
    elif searchtype == "Elser":
        query = build_elser_query(user_query)
    else:
        raise ValueError(f"Invalid searchtype: {searchtype}")

    results = es.search(index=byom_index_name, body=query, _source=True)

    # Set a default value for num_results
    num_results = 0

    if results and results.get('hits') and results['hits'].get('hits'):
        # Limit the number of results displayed

        num_results = min(5, len(results['hits']['hits']))

        for i in range(num_results):
            if results['hits']['hits'][i]["_source"].get("passages") and len(
                    results['hits']['hits'][i]["_source"]["passages"]) > 0:
                first_passage_text = results['hits']['hits'][i]["_source"]["passages"][0]["text"]
            else:
                first_passage_text = "No passages text available"

            print(first_passage_text)

    else:
        print("No results found.")

    return get_chat_guidance_rag(user_query, azureclient, results, conversation_history)


def search_products_v2(es, user_query, searchtype, rrf_rank_constant, rrf_window_size):
    # Select the appropriate query building function based on searchtype
    if searchtype == "Vector":
        query = build_knn_query(user_query, build_vector(es, user_query))
    elif searchtype == "BM25":
        query = build_bm25_query(user_query)
    elif searchtype == "Reciprocal Rank Fusion":
        query = build_rrf_query(build_vector(es, user_query), user_query, rrf_rank_constant, rrf_window_size)
    elif searchtype == "Elser":
        query = build_elser_query(user_query)
    else:
        raise ValueError(f"Invalid searchtype: {searchtype}")

    results = es.search(index=byom_index_name, body=query, _source=True)

    # Set a default value for num_results
    num_results = 0

    if results and results.get('hits') and results['hits'].get('hits'):
        # Limit the number of results displayed

        num_results = min(5, len(results['hits']['hits']))

        for i in range(num_results):
            if results['hits']['hits'][i]["_source"].get("passages") and len(
                    results['hits']['hits'][i]["_source"]["passages"]) > 0:
                first_passage_text = results['hits']['hits'][i]["_source"]["passages"][0]["text"]
            else:
                first_passage_text = "No passages text available"

            print(first_passage_text)

    else:
        print("No results found.")

    blog_bodies = []
    urls = []
    titles = []
    passages_texts = []
    scores = []

    # Check if there are any hits
    if results['hits']['hits']:
        # Process up to the first 3 results
        for hit in results['hits']['hits'][:3]:  # Limit to first 3 results
            source = hit["_source"]

            # Extract and accumulate the body content
            blog_body = source.get("body_content", "No body content available")
            blog_bodies.append(blog_body)

            # Extract and accumulate the URL, preferring 'additional_urls' if available and non-empty
            additional_urls = source.get("additional_urls", [])
            url = additional_urls[0] if additional_urls else source.get("url", "No URL available")
            urls.append(url)

            # Extract and accumulate the title
            title = source.get("title", "No title available")
            titles.append(title)

            # Extract and accumulate the first passage text if passages exist and are non-empty
            passages = source.get("passages", [])
            first_passage_text = passages[0].get("text",
                                                 "No passages text available") if passages else "No passages available"
            passages_texts.append(first_passage_text)

            # Extract and accumulate the score
            score = hit["_score"]
            scores.append(score)

    return blog_bodies
