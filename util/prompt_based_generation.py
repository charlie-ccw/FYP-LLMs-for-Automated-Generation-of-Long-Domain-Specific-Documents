from langchain_openai import ChatOpenAI
from globalParameter import parameters


def prompt_based_generation(prompt, model: str, temperature: float, json_format: bool = False):
    """
    Call model to answer your question
    :param prompt: Your complete prompt
    :param model: The name of model you want to use
    :param temperature: Higher scores indicate more random model outputs.
    :param json_format: True if you want the response to be Json format
    :return: model output based on your prompt and variables
    """
    # Set up the GPT connecction
    chat = ChatOpenAI(model=model, temperature=temperature)
    if json_format:
        chat = chat.with_structured_output(method="json_mode")
    # Call the model to response your prompt
    response = chat.invoke(prompt)

    return response


async def aprompt_based_generation(prompt, model: str, temperature: float, json_format: bool = False):
    """
    Call model to answer your question in Async way
    :param prompt: Your complete prompt
    :param model: The name of model you want to use
    :param temperature: Higher scores indicate more random model outputs.
    :param json_format: True if you want the response to be Json format
    :return: model output based on your prompt and variables
    """
    # Set up the GPT connecction
    chat = ChatOpenAI(model=model, temperature=temperature)
    if json_format:
        chat = chat.with_structured_output(method="json_mode")
    # Call the model to response your prompt in Async way
    response = await chat.ainvoke(prompt)

    return response
