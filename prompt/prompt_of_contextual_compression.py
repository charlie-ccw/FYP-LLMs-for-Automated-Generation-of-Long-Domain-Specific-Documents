CONTEXTUAL_COMPRESSION_SYSTEM = "You are an excellent contextual extractor"

CONTEXTUAL_COMPRESSION_PROMPT = """Given the following question and context, extract any part of the context *AS IS* that is relevant to answer the question. If none of the context is relevant return "NO OUTPUT STRING". 

Remember, *DO NOT* edit the extracted parts of the context.

> Question: {question}
> Context:
>>>
{context}
>>>
Extracted relevant parts:"""
