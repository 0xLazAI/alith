import os
from alith import Agent, WindowBufferMemory, ChromaDBStore, chunk_text
from pathlib import Path


def load_pdf_content(pdf_path: str) -> str:
    try:
        import PyPDF2
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text
    except ImportError:
        return "PyPDF2 not installed. Install with: pip install PyPDF2"
    except Exception as e:
        return f"Error reading PDF: {str(e)}"


def create_pdf_store(pdf_path: str):
    if not os.path.exists(pdf_path):
        print(f"PDF file {pdf_path} not found!")
        return None
    
    pdf_content = load_pdf_content(pdf_path)
    
    if pdf_content.startswith("Error") or pdf_content.startswith("PyPDF2"):
        print(pdf_content)
        return None
    
    docs = chunk_text(pdf_content)
    store = ChromaDBStore(path="./chroma_db", collection_name="pdf_chat")
    store.save_docs(docs)
    return store


def main():
    PDF_PATH = "path_to_your_pdf.pdf"
    API_KEY = os.getenv("GROQ_API_KEY")
    BASE_URL = "https://api.groq.com/openai/v1"
    
    store = create_pdf_store(PDF_PATH)
    if not store:
        print("Failed to create vector store. Exiting.")
        return
    
    agent = Agent(
        name="PDF Chat Assistant",
        model="llama-3.3-70b-versatile",
        api_key=API_KEY,
        base_url=BASE_URL,
        preamble="""You are a helpful assistant that can answer questions about the PDF document. 
        Use the provided document content to answer questions accurately. 
        If the information is not in the document, say so clearly.""",
        store=store,
        memory=WindowBufferMemory()
    )
    
    print("PDF loaded successfully! You can now chat with your document.")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        
        try:
            response = agent.prompt(user_input)
            print(f"Assistant: {response}\n")
        except Exception as e:
            print(f"Error: {str(e)}\n")


if __name__ == "__main__":
    main()
