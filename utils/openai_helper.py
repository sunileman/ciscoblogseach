import openai
import time

from variables import openai_completion_deployment_name, openai_api_sa_base

import streamlit as st
from openai import AzureOpenAI



def get_chat_guidance_rag(prompt, client, results, conversation_history):
    # Prepare the messages for Azure OpenAI, including the system message

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

    # Concatenate the body contents of up to the first 3 documents into a single blog_body field
    #blog_body = " ".join(blog_bodies)

    #print("Blog Bodies:", blog_body)

    st.session_state.messages.append({"role": "user", "content": f"Answer this question or comment: {prompt} using this text: {blog_body}. If the answer is not in the text, simply return NO ANSWER"})


    # Print the messages array for debugging or logging purposes
    # print("Messages being sent to Azure OpenAI:")
    # for message in messages:
    #     print(f"{message['role'].title()}: {message['content']}")

    # Generate response from Azure OpenAI
    response = client.chat.completions.create(
        model="gpt-35-turbo-16k",
        messages=st.session_state.messages
    )

    # Extract the text from the response
    return response.choices[0].message.content.strip()


def get_chat_guidance(client):
    # Prepare the messages for Azure OpenAI, including the system message


    # # Print the messages array for debugging or logging purposes
    print("Messages being sent to Azure OpenAI:")
    for message in st.session_state.messages:
       print(f"{message['role'].title()}: {message['content']}")

    # Generate response from Azure OpenAI
    response = client.chat.completions.create(
        model=openai_completion_deployment_name,
        messages=st.session_state.messages
    )


    # Extract the text from the response
    return response.choices[0].message.content.strip()



def get_openai_large_guidance(user_query, results, num_results, searchtype):
    client = AzureOpenAI(
        azure_endpoint=openai_api_sa_base,
        api_key=st.secrets['sa_pass'],
        api_version="2023-05-15"
    )

    processed_results = []
    retry_attempts = 3  # Number of retries before failing

    query_response_time = results['took']

    text = None
    url = None
    title = None
    first_passage_text = None

    for idx in range(num_results):
        genai_start_time = time.time()


        try:
            text = results['hits']['hits'][idx]["_source"]["body_content"]
            # Accessing the first value of 'additional_urls' if it exists and has at least one URL
            if results['hits']['hits'][idx]["_source"].get("additional_urls") and len(
                    results['hits']['hits'][idx]["_source"]["additional_urls"]) > 0:
                url = results['hits']['hits'][idx]["_source"]["additional_urls"][0]
            else:
                url = results['hits']['hits'][idx]["_source"]["url"]
            title = results['hits']['hits'][idx]["_source"]["title"]
            # Accessing the first instance of 'passages.text'
            # Check if 'passages' exists and has at least one item
            if results['hits']['hits'][idx]["_source"].get("passages") and len(
                    results['hits']['hits'][idx]["_source"]["passages"]) > 0:
                first_passage_text = results['hits']['hits'][idx]["_source"]["passages"][0]["text"]
            else:
                first_passage_text = "No passages text available"

        except KeyError:
            text = None
            url = None
            title = None

        score = results['hits']['hits'][idx]["_score"]

        for _ in range(retry_attempts):
            try:

                response = client.chat.completions.create(
                    model="gpt-35-turbo",  # model = "deployment_name".
                    messages=[
                        {"role": "system",
                         "content": "You are an AI assistant. Your answers should stay short and concise. explain your answer. no formalities."},
                        {"role": "user",
                         "content": f"Answer this question. Keep the response less than 30 words.  {user_query} based on the following text {text}"}
                    ]
                )

                genai_end_time = time.time()
                genai_query_time = (genai_end_time - genai_start_time) * 1000

                completion_output = response.choices[0].message.content

                processed_results.append(
                    (text, completion_output, score, query_response_time, genai_query_time, url, title, first_passage_text))
                break  # If successful, break out of the retry loop

            except openai.error.RateLimitError as e:
                # Handle rate limit error
                if _ < retry_attempts - 1:  # If it's not the last attempt
                    print(f"Rate Limit Error: {e}. Retrying in 1 second...")
                    time.sleep(1)  # Wait for 1 second before retrying
                else:
                    print(f"Rate Limit Error: {e}. No more retries.")

            except openai.error.APIError as e:
                print(f"OpenAI API returned an API Error: {e}")
                break

            except openai.error.AuthenticationError as e:
                print(f"OpenAI API returned an Authentication Error: {e}")
                break

            except openai.error.APIConnectionError as e:
                print(f"Failed to connect to OpenAI API: {e}")
                break

            except openai.error.InvalidRequestError as e:
                print(f"Invalid Request Error: {e}")
                break

            except openai.error.ServiceUnavailableError as e:
                print(f"Service Unavailable: {e}")
                break

            except openai.error.Timeout as e:
                print(f"Request timed out: {e}")
                break

            except:
                # Handles all other exceptions
                print("An unexpected exception has occurred.")
                break

            time.sleep(4)

    return processed_results, results
