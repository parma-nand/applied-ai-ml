"""
semantic_similarity.py
-----------------------
Semantic similarity using LangChain + all-MiniLM-L6-v2
Paste your resume text directly into RESUME_TEXT below.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ─────────────────────────────────────────────
#  1. PASTE YOUR RESUME TEXT HERE
# ─────────────────────────────────────────────

RESUME_TEXT = """
Parma Nand
Associate Consultant at Capgemini, Pune
B.Tech - NIT Jalandhar (2022)

Skills: Python, FastAPI, LangChain, Docker, Airflow, spaCy,
        Pinecone, ChromaDB, sentence-transformers, REST APIs

Experience:
- Built Resume Intelligence Platform with PDF parsing pipeline,
  vector embeddings, and semantic job matching using Pinecone.
- Developed NLP pipeline: PyMuPDF extraction, spaCy NER,
  LangChain chunking, Pydantic schemas.
- Automation testing background with 4 years at Capgemini.
- Hands-on with LLMs, RAG pipelines, and agentic AI workflows.

Projects:
- Resume Job Matcher: FastAPI + Streamlit + Docker Compose
- NLP Pipeline: section detection, skill categorization, embeddings
"""

# ─────────────────────────────────────────────
#  2. JOB DESCRIPTIONS TO MATCH AGAINST
# ─────────────────────────────────────────────

JOB_DESCRIPTIONS = [
    {
        "id": "JD1",
        "title": "LLM Application Developer",
        "text": "Build LLM-powered apps using LangChain, RAG pipelines, "
                "Pinecone, ChromaDB. Python, FastAPI, OpenAI or open-source models.",
    },
    {
        "id": "JD2",
        "title": "Senior ML Engineer",
        "text": "Production ML with PyTorch, TensorFlow, MLflow. "
                "Feature pipelines, model deployment, Python.",
    },
    {
        "id": "JD3",
        "title": "Data Engineer",
        "text": "ETL pipelines with Apache Spark, Airflow, Snowflake, BigQuery. "
                "Strong SQL, AWS or GCP.",
    },
    {
        "id": "JD4",
        "title": "Backend Engineer",
        "text": "REST APIs with FastAPI or Django, Docker, Kubernetes, PostgreSQL.",
    },
    {
        "id": "JD5",
        "title": "NLP Research Engineer",
        "text": "BERT, GPT, T5 models. HuggingFace Transformers, NER, "
                "text classification, summarization.",
    },
]


# ─────────────────────────────────────────────
#  3. LOAD EMBEDDING MODEL
# ─────────────────────────────────────────────

print("Loading all-MiniLM-L6-v2 ...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)
print("Model loaded.\n")


# ─────────────────────────────────────────────
#  4. CHUNK THE RESUME
# ─────────────────────────────────────────────

splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
resume_chunks = splitter.create_documents([RESUME_TEXT])

print(f"Resume split into {len(resume_chunks)} chunks:")
for i, chunk in enumerate(resume_chunks):
    print(f"  Chunk {i+1}: {chunk.page_content[:80].strip()} ...")
print()


# ─────────────────────────────────────────────
#  5. BUILD FAISS VECTOR STORE FROM RESUME
# ─────────────────────────────────────────────

vectorstore = FAISS.from_documents(resume_chunks, embeddings)
print("FAISS index built from resume chunks.\n")


# ─────────────────────────────────────────────
#  6. MATCH EACH JOB AGAINST THE RESUME
# ─────────────────────────────────────────────

print("=" * 55)
print("  JOB MATCH RESULTS (higher score = better match)")
print("=" * 55)

results = []

for jd in JOB_DESCRIPTIONS:
    # similarity_search_with_score returns (Document, L2 distance)
    # Lower L2 distance = more similar (we convert to a % score)
    docs_scores = vectorstore.similarity_search_with_score(jd["text"], k=1)
    best_doc, l2_distance = docs_scores[0]

    # Convert L2 distance to a 0-100 similarity score
    # all-MiniLM-L6-v2 embeddings are normalized, so L2 in [0, 2]
    similarity_pct = round((1 - l2_distance / 2) * 100, 1)

    results.append({"title": jd["title"], "score": similarity_pct, "chunk": best_doc.page_content.strip()})

# Sort by score descending
results.sort(key=lambda x: x["score"], reverse=True)

for rank, r in enumerate(results, 1):
    bar = "█" * int(r["score"] / 5) + "░" * (20 - int(r["score"] / 5))
    print(f"\n  [{rank}] {r['title']}")
    print(f"      Score : {bar} {r['score']}%")
    print(f"      Matched chunk: \"{r['chunk'][:90]} ...\"")

print("\n" + "=" * 55)
print("  Top match →", results[0]["title"])
print("=" * 55)