"""
This Python file provides a custom model API call method that can perform text extraction tasks
from PDF-converted images based on your input.
"""
import json

from openai import AsyncOpenAI
import base64


# Function to encode the image
def encode_image(image_path):
    """
    To convert picture into correct format
    :param image_path: The path of picture you want to use
    :return: formated picture code
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


async def aget_openai_response(messages, section_id: str, section_name: str,
                               model: str = "gpt-4o", temperature: float = 0.5,
                               json_format: bool = False, retry_num: int = 0):
    """
    To Extract info of picture based on your message prompt and return in JSON format
    :param messages: Your Message prompt which will send to model directly
    :param section_id: The id of section you are trying to extract
    :param section_name: The name of section you are trying to extract
    :param model: The name of model you want to use
    :param temperature: Higher scores indicate more random model outputs.
    :param json_format: True if you want the response to be Json format, Recommend setting to True.
    :param retry_num: The number of retry attempts you prefer when the model extraction fails.
    :return: Extraced info of section
    """
    # Set the client for connection
    client = AsyncOpenAI()
    # Set the output format
    response_format = {"type": "json_object"} if json_format else {"type": "text"}
    # Initialise the variables
    try_count = 0
    output = None
    # Start to call Model API
    while try_count <= retry_num:
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                response_format=response_format
            )
            output = json.loads(response.choices[0].message.content)
        except Exception as e:
            try_count += 1
            print(e)
            if try_count <= retry_num:
                print(f"retry {section_id} {section_name} --- {try_count}/{retry_num}")
        else:
            break

    # The default response when the model is unable to complete the task.
    if output is None:
        output = {
            section_id: {
                "section_id": section_id,
                "section_name": section_name,
                "section_info": None,
            }
        }
        return output
    else:
        return output

