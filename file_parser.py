import fitz
from langchain.docstore.document import Document


def pdf_to_langchain_docs(pdf_path):
    """parses a .pdf file.
        Inputs:
            pdf_path: The path to the pdf file.
        Outputs:
            langchain_docs: A list of langchain documents. Each document is a page from the pdf file.
    """
    # Open the PDF file
    pdf_document = fitz.open(pdf_path)
    langchain_docs = []
    
    # Iterate through each page in the PDF
    for page_num in range(pdf_document.page_count):
        page = pdf_document[page_num]
        page_content = page.get_text()  # Get the text content of the page

        # Create a LangChain Document for each page
        doc = Document(
            page_content=page_content,
            metadata={
                "page_number": page_num + 1
            }
        )
        
        langchain_docs.append(doc)
    
    pdf_document.close()  # Close the PDF file
    return langchain_docs

# pdf_file = "First_Chat_Bot\Thesis.pdf"
# documents = pdf_to_langchain_docs(pdf_path=pdf_file)
# print(documents)