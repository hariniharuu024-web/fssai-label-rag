FSSAI Packaging & Labeling Compliance RAG
Stack: LangChain + Qdrant Cloud + Gemini + FastAPI (deploy on Render)

Run order:
1. pip install -r requirements.txt
2. Fill keys in .env
3. Put labelling.pdf inside data/
4. python ingest.py
5. uvicorn main:app --reload
