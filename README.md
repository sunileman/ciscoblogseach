# Cisco Blog Search


## Apps

### Chat Bot
```commandline
streamlit run cisco-chatbot.py
```

### Blog Search
```commandline
streamlit run cisco-blog-search.py
```

### Update variables

`.streamlit/secrets.toml`
```
pass = "xxxxxxx"
sa_pass = "xxxxx" ##used for large token model

es_username = 'xxxxx'
es_password = 'xxxxx'
es_cloudid = 'xxxx'
```


'variables.py'
```commandline
openai_api_base = "https://xxxxx.openai.azure.com"
openai_api_sa_base = "https://xxxx.openai.azure.com"
```

