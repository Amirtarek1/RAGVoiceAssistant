# # pip install -U langchain-community

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import openai 
from dotenv import load_dotenv
import shutil

# File path (raw string for Windows)
Data = r"C:\Centraly_Project\Data\info.md"

def load_data():
    loader = TextLoader(Data, encoding="utf-8")  
    documents = loader.load()
    return documents

def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    
    # Check first chunk
    document = chunks[0]
    print("First chunk content:\n", document.page_content)
    print("Metadata:\n", document.metadata)
    
    return chunks


def save_to_chroma(chunks: list[Document]):
    # Clear out the database first.
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Create a new DB from the documents.
    db = Chroma.from_documents(
        chunks, OpenAIEmbeddings(), persist_directory=CHROMA_PATH
    )
    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")


if __name__ == "__main__":
    main()

# docs = load_data()
# chunks = split_text(docs)


# # Show first 5 chunks
# for i, chunk in enumerate(chunks[:5], start=1):
#     print(f"\n--- Chunk {i} ---")
#     print(chunk.page_content)
#     print("Metadata:", chunk.metadata)


