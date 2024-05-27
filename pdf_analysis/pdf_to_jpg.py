"""
Before we use the multimodal model to extract text, we need to convert the content of each page into image format.
"""
import os

from pdf_analysis.utils import pdf_to_images

# Set the folder which contains golden standard PDF files
folder_70 = "../file/Energy_demand/structure_1"
# Get all files
files = os.listdir(folder_70)
# Set the path for storing pictures
output_path = "../file/Energy_demand_picture/structure_1"

# Start iteration and get pictures for each single page of each file and store them into correct path for Extraction process
for file in files:
    print(file)
    file_path = os.path.join(folder_70, file)
    pdf_to_images(file_path, output_path, dpi=400)
