# LLMs for Automated Generation of Long Domain Specific Documents
## Project Overview

This project explores the potential of large language models (LLMs) in generating long, domain-specific documents. It addresses the limitations of traditional models in handling extensive contexts, incorporating domain-specific knowledge, and maintaining coherence over lengthy texts. Techniques such as Retrieval-Augmented Generation (RAG), prompt engineering with chains of thought, and AI Agent frameworks are integrated to enhance performance.

## Installation

To get started, clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/charlie-ccw/FYP-LLMs-for-Automated-Generation-of-Long-Domain-Specific-Documents.git
cd FYP-LLMs-for-Automated-Generation-of-Long-Domain-Specific-Documents
pip install -r requirements.txt
```

#### Use the following link to download All Files: [Download the large file here](https://drive.google.com/drive/folders/1baKAXYJAMIiHxPgSQfqYo9-sKRPGBlFb?usp=sharing)

## Set Up

1. **Get your OPENAI_API_KEY**: Obtain your OpenAI API key and fill it into `globalParameter/parameters.py`.

2. **(Optional) Get your LangSmith account**: If you have a LangSmith account, fill in the key and project info into `globalParameter/parameters.py`.

3. **Choose a Chat Model**: Select a chat model from the [OpenAI Web](https://platform.openai.com/docs/models) and fill in the model name into `globalParameter/parameters.py`.

4. **Select a Domain Name**: Choose a domain name and fill it into `globalParameter/parameters.py`.

## Usage of each Python Script

### pdf_analysis/pdf_content_get.py
This script utilizes the `pdfplumber` Python library to extract text information from each page of PDF files. It performs regular expression matching to verify the presence of queried table of contents names on each page, records page numbers for matched names, and iterates until all names are matched or the entire file has been processed. Errors and unmatched pages will be printed and need to be manually corrected to ensure accuracy.

### pdf_analysis/pdf_to_jpg.py
This script employs the `pdf2image` Python library to convert PDF pages into images. The conversion is done at a DPI setting of 400 to ensure high image quality, which is crucial for subsequent text and table extraction processes.

### pdf_analysis/pdf_text_extract.py
This tool focuses on reading text and extracting tables from PDFs using a multimodal model approach for more accurate and structured content extraction. Errors and unmatched pages will be printed and need to be manually corrected to ensure accuracy.

### generation_version_1.py (Baseline)
This baseline version uses a straightforward approach for document generation without sophisticated retrieval or refinement methods. It primarily serves as a reference point for evaluating improvements in subsequent versions.

### generation_version_2.py (Retrieval Question-Answer Tool Only)
This version integrates a retrieval-based question-answer tool that leverages information retrieval techniques to fetch relevant content fragments from a knowledge base, enhancing the response quality by providing contextually appropriate information.

### generation_version_3.py (Retrieval Refine Tool Only)
This version introduces a retrieval refine tool that optimizes draft texts based on historical document content retrieved from the knowledge base. It improves language, structure, and overall content quality by leveraging retrieved examples.

### generation_version_4.py (Retrieval QA with LLM and Resort Tool Only)
This version enhances the retrieval QA tool by incorporating LLM capabilities and a resorting mechanism. It includes multi-query retrieval, hierarchical retrieval, and secondary re-ranking to obtain comprehensive and accurate knowledge segments, further refined for improved relevance and coherence.

### generation_version_5.py (AI Agent With Retrieval QA with LLM and Resort Tool)
The most advanced version, incorporating an AI agent that utilizes the retrieval QA tool with LLM and resort capabilities. This version integrates multiple advanced techniques, including prompt engineering, chains of thought, and retrieval-augmented generation, to handle complex document generation tasks more effectively.

### evaluation.py
This script evaluates the generated documents using various metrics such as BLEU, ROUGE, and METEOR. It measures the accuracy, fluency, and relevance of the generated text compared to reference texts, providing a comprehensive assessment of the document generation performance.

## Generation Steps
1. Choos your Model Name and Domain Name, fill them into `globalParameter/parameters.py`.
2. use `generation_version_5.py` or others (version_1, version_2, version_4) to generate the draft version.
3. use `generation_version_3.py` to generate the final refined version. NOTE: you need to set `base_version` in `generation_version_3.py`.

## Project Template Web
https://view.officeapps.live.com/op/view.aspx?src=https%3A%2F%2Fverra.org%2Fwp-content%2Fuploads%2F2024%2F04%2FVCS-Project-Description-Template-v4.4-FINAL2.docx&wdOrigin=BROWSELINK

