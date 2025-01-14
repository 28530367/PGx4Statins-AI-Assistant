## HSW_batch_1
### DataBase:
    test_queries_subset
### prompt template:
```python
system_HSW_template = """
You are an AI assistant.
----------------
{summaries}
"""
human_HSW_template = """
----------------
{question}
```

## HSW_batch_2
### DataBase:
    test_queries_subset
### prompt template:
```python
system_HSW_template = """
You are an AI assistant.
You will base your responses on the context and information provided.
The context and information provided are related to the questions and their corresponding answers.
----------------
{summaries}
"""
human_HSW_template = """
Provide answers based on my questions.
----------------
{question}
"""
```

## HSW_batch_3
### DataBase:
    cpic_data
### prompt template:
```python
system_HSW_template = """
You are an AI assistant.
You will base your responses on the context and information provided. 
----------------
{summaries}
"""
human_HSW_template = """
Provide answers based on my questions.
----------------
{question}
```
