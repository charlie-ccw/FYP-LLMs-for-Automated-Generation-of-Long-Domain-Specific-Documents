RETRIEVAL_QA_TOOL_SYSTEM = "You are a useful assistant."

RETRIEVAL_QA_TOOL_PROMPT = """Given the following extracted parts of long documents and a question, create a final answer. 
If you don't know the answer, just say that you don't know. Don't try to make up an answer.
Note: respond in JSON format:
{{
    'answer': 'The answer you find based on extracted parts of long documents, try your best to refine the answer',
    'find_answer_in_extracted_part': 'YES: If you can find an answer from extracted parts; NO: if you cannot find an answer from extracted parts',
}}

QUESTION: {question}
=========
{summaries}
=========
FINAL ANSWER:
your answer in correct json format"""