import os, time
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_qdrant import QdrantVectorStore

load_dotenv()
app = FastAPI(title="FSSAI Label Compliance RAG")

emb = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    output_dimensionality=768,
)
store = QdrantVectorStore.from_existing_collection(
    embedding=emb,
    collection_name="fssai_labels",
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    timeout=120,
)

MODELS = ["gemini-3.8-flash", "gemini-3.6-flash", "gemini-3.1-flash-lite", "gemini-flash-latest"]
llms = [ChatGoogleGenerativeAI(model=m, temperature=0, max_retries=1) for m in MODELS]


class Label(BaseModel):
    text: str


PROMPT = """You are an FSSAI labeling compliance checker.
Using ONLY the regulation excerpts below, check the product label text.
Give your answer under these headings:
1) Missing mandatory declarations
2) Allergen issues
3) Ingredient / additive violations
4) What is correct
If the excerpts are not enough to decide something, say so. Mention page numbers.

REGULATION EXCERPTS:
{context}

LABEL TEXT:
{label}
"""


@app.post("/validate")
def validate(label: Label):
    docs = store.similarity_search(label.text, k=5)
    docs += store.similarity_search(
        "mandatory label declarations allergen ingredients list", k=4)
    ctx = "\n\n".join(
        f"[page {d.metadata.get('page')}] {d.page_content}" for d in docs)
    pages = sorted({str(d.metadata.get("page")) for d in docs})
    prompt = PROMPT.format(context=ctx, label=label.text)

    for attempt in range(3):
        for m in llms:
            try:
                res = m.invoke(prompt)
                text = res.content
                if isinstance(text, list):
                    text = "".join(
                        p.get("text", "") if isinstance(p, dict) else str(p)
                        for p in text)
                return {"result": text, "sources": pages}
            except Exception as e:
                print("Model failed:", str(e)[:120])
        time.sleep(4)

    return {
        "result": "The AI model is busy right now. Here are the FSSAI rules "
                  "retrieved for this label:\n\n" + ctx[:3500],
        "sources": pages,
    }


@app.get("/", response_class=HTMLResponse)
def home():
    with open("index.html", encoding="utf-8") as f:
        return f.read()