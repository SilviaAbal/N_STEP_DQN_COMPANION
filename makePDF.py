import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, PageBreak, Image

# Input folder containing images
input_folder = 'figs/recipes'

# Output PDF file
output_pdf = 'figs/output.pdf'

# Number of images per page
images_per_page = 5

# Create a list of image files in the folder
image_files = sorted([f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.png', '.jpeg', '.gif'))])

# Calculate the size of each image on the page
page_width, page_height = letter
margin = 36  # Margin size in points (1/2 inch)

image_width = page_width - 2 * margin
image_height = (page_height - 2 * margin) / (images_per_page+1)

# Create a PDF document
doc = SimpleDocTemplate(output_pdf, pagesize=letter, leftMargin=margin, rightMargin=margin, topMargin=margin, bottomMargin=margin)

# Initialize variables for image positioning
images = []

# Loop through image files and add them to the PDF
for image_file in image_files:
    # Open and resize the image
    image_path = os.path.join(input_folder, image_file)
    img = Image(image_path, width=image_width, height=image_height)

    # Add the image to the list of images on the page
    images.append(img)

    # If enough images are collected for one page, add a page and reset images list
    if len(images) == images_per_page:
        images.append(PageBreak())  # Add a page break for the next set of images

doc.build(images)

print(f'PDF created with {len(image_files)} images: {output_pdf}')
