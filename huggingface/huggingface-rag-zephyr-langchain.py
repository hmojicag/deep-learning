# python -m venv .venv
# source .venv/bin/activate
# .\.venv\Scripts\Activate.ps1
# pip install langchain langchain-community langchain-text-splitters torch transformers accelerate bitsandbytes transformers sentence-transformers
# pip install faiss-gpu

# export GITHUB_PERSONAL_ACCESS_TOKEN="ghp_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
# python .\huggingface\huggingface-rag-zephyr-langchain.py
# python huggingface/huggingface-rag-zephyr-langchain.py


from getpass import getpass
from langchain_community.document_loaders import GitHubIssuesLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

ACCESS_TOKEN = getpass("YOUR_GITHUB_PERSONAL_TOKEN")

if __name__ == "__main__":
    loader = GitHubIssuesLoader(
        repo="huggingface/peft",
        access_token=ACCESS_TOKEN,
        include_prs=False,
        state="all"
    )

    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=30)

    chunked_docs = splitter.split_documents(docs)

    # I moved to use a jupyter notebook after this
    # ------------------------------------------------------------------------------------------------------------------

    db = FAISS.from_documents(chunked_docs, HuggingFaceEmbeddings(model_name='BAAI/bge-base-en-v1.5'))

