MULTI_QUERY_SYSTEM = "You are an AI language model assistant."

MULTI_QUERY_PROMPT = """Your task is to generate five 
different versions of the given user question to retrieve relevant documents from a vector 
database. By generating multiple perspectives on the user question, your goal is to help
the user overcome some of the limitations of the distance-based similarity search. 
Provide these alternative questions separated by newlines.
Original question: {question}

Note: output in JSON format:
{{
    "query_1": Your query 1,
    "query_2": Your query 2,
    "query_3": Your query 3,
    "query_4": Your query 4,
    "query_5": Your query 5
}}"""