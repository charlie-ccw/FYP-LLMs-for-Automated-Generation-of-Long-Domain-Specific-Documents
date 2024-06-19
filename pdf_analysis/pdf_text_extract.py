"""
By using different PDF content extraction tools, we ultimately chose to use a multimodal model for text extraction.
This approach not only allows for the complete retrieval of content but also removes noise information such as headers and footers.
Additionally, the large model will extract the complete content in the correct format through self-reflection,
including both text and table information.
Note:
    Due to the inherent limitations of the large model and the potential for model hallucinations,
    some complex sections may not be extracted correctly automatically.
    All run results will be printed in the log file, and manual extraction will be required based on ERROR info.
"""
import asyncio
import os
import json
import logging

from globalParameter.parameters import DOMAIN
from util.openai_api_for_pdf_extract import encode_image, aget_openai_response

# Set up the logging
logging.basicConfig(level=logging.WARNING,
                    filename='pdf_text_extract.log',
                    filemode='a',
                    format='%(asctime)s - %(levelname)s: %(message)s')


# Define the main program
async def main():
    # Set the picture folder (contains the picture of each single page) and content folder (contains the correct content info)
    picture_folder = f"../file/{DOMAIN}_picture/structure_1"
    content_folder = f"../file/{DOMAIN}_content/structure_1"
    # Get full files inside content folder
    files = os.listdir(content_folder)

    # Start the iteration and extract info from each file
    for idx, file in enumerate(files):
        file_data = {}
        # Get the file name without extension
        file_name_without_extension = os.path.splitext(file)[0]
        print(f"-------------------- {file_name_without_extension} --------------------")
        # Record the progress
        logging.warning(
            f"\n\n\n>>>>>>>>>>>>>>>>>>>> Trying to extract data from file: '{file_name_without_extension}' <<<<<<<<<<<<<<<<<<<< {idx}/{len(files)}")
        # Get the correct content of current file
        with open(f'{content_folder}/{file_name_without_extension}.json', 'r', encoding='utf-8') as f:
            file_content = json.load(f)

        # Get section ids from content and sort ids in correct order
        section_ids = list(file_content.keys())
        sorted_section_ids = sorted(section_ids,
                                    key=lambda x: [int(part) if part.isdigit() else float('inf') for part in
                                                   x.split('.')])

        # Start to extract info of each section
        section_messages = []
        sections_ids_in_messages = []
        # Start section iteration
        for i, section_id in enumerate(sorted_section_ids[:-1]):
            # Get the correct page num range for current section
            page_number_start = int(file_content[section_id]["page_number"])
            page_number_end = int(file_content[sorted_section_ids[i + 1]]["page_number"])

            # Get correct section name
            section_name = file_content[section_id]["section_name"]

            # Get the image of the corresponding page
            page_pictures = []
            for page_i in range(page_number_start, page_number_end + 1):
                page_pictures.append(os.path.join(picture_folder, file_name_without_extension, f"page_{page_i}.jpg"))

            # prepare the messages for current section id
            prompt = f"""Display all the contents for section {section_id} {section_name} in the correct format.
    Note: The output should be in JSON format:
    "1.1": {{
        "section_id": "1.1",
        "section_name": "name of section 1.1",
        "section_info": "whole info of section 1.1 in string format with good structure, Note: in string format",
    }}"""
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                    ]
                }
            ]
            # Enhance the message by adding image content
            for page_picture in page_pictures:
                base64_image = encode_image(page_picture)
                messages[0]["content"].append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                    },
                })

            # store the messages for async call to speed up the extraction process
            section_messages.append(messages)
            sections_ids_in_messages.append(section_id)

            # start to call openai api in async way, every 5 sections
            if len(section_messages) == 5:
                logging.warning(sections_ids_in_messages)
                tasks = []
                for messages_id, messages in enumerate(section_messages):
                    # Create tasks for async run
                    tasks.append(aget_openai_response(section_id=sections_ids_in_messages[messages_id],
                                                      section_name=file_content[sections_ids_in_messages[messages_id]][
                                                          'section_name'],
                                                      messages=messages, temperature=0.5, json_format=True, retry_num=1))
                # Get responses
                responses = await asyncio.gather(*tasks)
                # Check the format of response and add them into logging if error happens
                for response in responses:
                    # If the model doesn't output response with correct Json format
                    if len(list(response.keys())) > 1:
                        logging.error(
                            f"---------- PLEASE CHECK THE DATA OF THIS FILE ---------- response length over 1: {response.keys()}")
                    # If the model output doesn't contain section info
                    for key, value in response.items():
                        file_data[key] = value
                        if value['section_info'] is None:
                            logging.error(
                                f"---------- PLEASE CHECK THE DATA OF THIS FILE ---------- FAIL to process: {value['section_id']} {value['section_name']}")
                section_messages = []
                sections_ids_in_messages = []

        # check if there are sections left, we need to run async call for one more time
        if len(section_messages) > 0:
            logging.warning(sections_ids_in_messages)
            tasks = []
            for messages_id, messages in enumerate(section_messages):
                tasks.append(aget_openai_response(section_id=sections_ids_in_messages[messages_id],
                                                  section_name=file_content[sections_ids_in_messages[messages_id]][
                                                      'section_name'],
                                                  messages=messages, temperature=0.5, json_format=True, retry_num=1))
            responses = await asyncio.gather(*tasks)
            for response in responses:
                if len(list(response.keys())) > 1:
                    logging.error(
                        f"---------- PLEASE CHECK THE DATA OF THIS FILE ---------- response length over 1: {response.keys()}")

                for key, value in response.items():
                    file_data[key] = value
                    if value['section_info'] is None:
                        logging.error(
                            f"---------- PLEASE CHECK THE DATA OF THIS FILE ---------- FAIL to process: {value['section_id']} {value['section_name']}")

        # check the content of file data
        for key, value in file_data.items():
            # Check if the section id of model response is correct
            try:
                sorted_section_ids.remove(key)
            except Exception:
                logging.error(f"---------- PLEASE CHECK THE DATA OF THIS FILE ---------- section id error {key}")

            # Check if the section name of model response is correct
            if value['section_name'].lower() == file_content[key]['section_name'].lower():
                file_data[key]['section_name'] = file_content[key]['section_name']
            else:
                logging.error(
                    f"---------- PLEASE CHECK THE DATA OF THIS FILE ---------- section name error {value['section_name']}")
        # Check if there is any unprocessed sections
        if not sorted_section_ids == ["END"]:
            logging.error(
                f"---------- PLEASE CHECK THE DATA OF THIS FILE ---------- section not processed {sorted_section_ids}")

        # store the file data
        os.makedirs(f"../file/{DOMAIN}_extract/structure_1", exist_ok=True)
        with open(f'../file/{DOMAIN}_extract/structure_1/{file_name_without_extension}.json', 'w', encoding='utf-8') as f:
            json.dump(file_data, f, indent=4)


if __name__ == "__main__":
    asyncio.run(main())
