import os, glob, time
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore

load_dotenv()

pdfs = glob.glob("data/*.pdf")
if not pdfs:
    raise SystemExit("No PDF found in data/ folder. Put labelling.pdf there first.")

docs = []
for f in pdfs:
    print("Reading", f)
    docs += PyPDFLoader(f).load()

chunks = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=150).split_documents(docs)
print("Total chunks:", len(chunks))

# Threshold / topic tagging (W3-4 of sir's plan)
for c in chunks:
    t = c.page_content.lower()
    tags = []
    if "allergen" in t or "allergy" in t:
        tags.append("allergen")
    if "additive" in t or "ins " in t:
        tags.append("additive")
    if "per cent" in t or "%" in t:
        tags.append("threshold")
    if "warning" in t or "mandatory" in t or "shall" in t:
        tags.append("mandatory")
    c.metadata["tags"] = tags

emb = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    output_dimensionality=768,   # smaller vectors = smaller upload
)
kw = dict(url=os.getenv("QDRANT_URL"),
          api_key=os.getenv("QDRANT_API_KEY"),
          collection_name="fssai_labels",
          timeout=120)

BATCH = 10
store = QdrantVectorStore.from_documents(
    chunks[:BATCH], emb, force_recreate=True, **kw)
print("Stored", min(BATCH, len(chunks)))
for i in range(BATCH, len(chunks), BATCH):
    store.add_documents(chunks[i:i + BATCH])
    print("Stored", min(i + BATCH, len(chunks)))
    time.sleep(8)  # avoid free-tier rate limit
print("DONE. Ingested", len(chunks), "chunks into Qdrant.")
