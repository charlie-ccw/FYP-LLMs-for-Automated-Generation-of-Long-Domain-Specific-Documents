"""
Before we use the multimodal model to extract text, we need to convert the content of each page into image format.
"""
import os

from globalParameter.parameters import DOMAIN
from pdf_analysis.utils import pdf_to_images

# Set the folder which contains golden standard PDF files
folder = f"../file/{DOMAIN}/structure_1"
# Get all files
files = os.listdir(folder)
# Set the path for storing pictures
output_path = f"../file/{DOMAIN}_picture/structure_1"

# Start iteration and get pictures for each single page of each file and store them into correct path for Extraction process
for file in files:
    print(file)
    file_path = os.path.join(folder, file)
    pdf_to_images(file_path, output_path, dpi=400)
