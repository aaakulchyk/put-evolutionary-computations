import nbformat
from nbconvert import PDFExporter
from traitlets.config import Config


def convert_notebook_to_pdf(notebook_path, output_pdf_path):
    """
    Convert the current Jupyter notebook to PDF, ignoring code cells and keeping markdown and outputs.

    Parameters:
    notebook_path (str): The path to the .ipynb notebook.
    output_pdf_path (str): The path where the PDF will be saved.
    """
    # Load the notebook
    with open(notebook_path, "r", encoding="utf-8") as f:
        notebook_content = nbformat.read(f, as_version=4)

    # Create a custom configuration to remove code cells
    c = Config()
    c.TemplateExporter.exclude_input = True  # Exclude the input (code) cells
    c.TemplateExporter.exclude_input_prompt = True  # Exclude input prompts

    # Create a PDF exporter with the custom config
    pdf_exporter = PDFExporter(config=c)

    # Convert the notebook to PDF
    pdf_data, _ = pdf_exporter.from_notebook_node(notebook_content)

    # Save the output PDF
    with open(output_pdf_path, "wb") as pdf_file:
        pdf_file.write(pdf_data)
