import os

# Your openai key for API Call
os.environ["OPENAI_API_KEY"] = "your openai keys"

# Set up the langsmith tool
os.environ["LANGCHAIN_TRACING_V2"] = ""
os.environ["LANGCHAIN_PROJECT"] = f""
os.environ["LANGCHAIN_ENDPOINT"] = ""
os.environ["LANGCHAIN_API_KEY"] = ""

# Set up your model type
MODEL = "gpt-3.5-turbo"
