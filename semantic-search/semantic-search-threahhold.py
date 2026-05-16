"""
rag_pipeline.py
---------------
RAG pipeline with threshold-based resume screening.
Simulates how real ATS systems (LinkedIn, Naukri, Greenhouse)
filter candidates using similarity score thresholds.

Flow:
  RESUME_TEXT  →  chunk  →  embed  →  ChromaDB
  JD query     →  embed  →  similarity search  →  top 3 chunks
                                                →  threshold check
                                                →  SHORTLISTED / REJECTED
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ─────────────────────────────────────────────────────
#  THRESHOLD CONFIG  ← tune these like a real ATS
# ─────────────────────────────────────────────────────

SHORTLIST_THRESHOLD = 70.0   # >= 70%  → SHORTLISTED    (strong match)
REVIEW_THRESHOLD    = 50.0   # >= 50%  → MANUAL REVIEW  (partial match)
                             #  < 50%  → REJECTED        (weak match)

# ─────────────────────────────────────────────────────
#  1. PASTE YOUR RESUME TEXT HERE
# ─────────────────────────────────────────────────────

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
- Semantic Search: all-MiniLM-L6-v2, FAISS, ChromaDB integration

Education:
- B.Tech, NIT Jalandhar, 2022
- AI/ML course (ongoing): classical ML, deep learning, LLMs, PySpark

Certifications:
- Python for Data Science
- LangChain and LLM application development
"""

# ─────────────────────────────────────────────────────
#  2. JOB DESCRIPTIONS  (the queries in ATS screening)
# ─────────────────────────────────────────────────────

JOB_DESCRIPTIONS = [
    {
        "role":    "LLM App Developer",
        "company": "Sarvam AI",
        "query":   "LangChain RAG pipeline Pinecone ChromaDB FastAPI LLM Python",
    },
    {
        "role":    "Applied AI Engineer",
        "company": "Observe.AI",
        "query":   "NLP spaCy transformers HuggingFace text classification Python",
    },
    {
        "role":    "Backend Engineer",
        "company": "Razorpay",
        "query":   "REST API FastAPI Django PostgreSQL Docker Kubernetes microservices",
    },
    {
        "role":    "Data Engineer",
        "company": "Flipkart",
        "query":   "Apache Spark Airflow ETL pipeline SQL BigQuery Kafka PySpark",
    },
    {
        "role":    "iOS Developer",           # intentionally weak match
        "company": "Some Startup",
        "query":   "Swift SwiftUI iOS mobile app Xcode UIKit CoreData",
    },
]

SEP  = "-" * 60
SEP2 = "=" * 60

# ─────────────────────────────────────────────────────
#  3. LOAD EMBEDDING MODEL
# ─────────────────────────────────────────────────────

print(SEP2)
print("  Loading all-MiniLM-L6-v2 ...")
print(SEP2)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)
print("  Model ready.\n")

# ─────────────────────────────────────────────────────
#  4. CHUNK THE RESUME
# ─────────────────────────────────────────────────────

splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks   = splitter.create_documents([RESUME_TEXT])

print(f"  Resume split into {len(chunks)} chunks:")
for i, c in enumerate(chunks):
    print(f"    Chunk {i+1}: {c.page_content[:65].strip()} ...")
print()

# ─────────────────────────────────────────────────────
#  5. BUILD CHROMADB VECTOR STORE
# ─────────────────────────────────────────────────────

print(SEP2)
print("  Building FAISS index ...")
print(SEP2)

vectorstore = FAISS.from_documents(
    documents=chunks,
    embedding=embeddings,
)
print("  FAISS index ready (in-memory)\n")

# ─────────────────────────────────────────────────────
#  6. HELPER — verdict based on threshold
# ─────────────────────────────────────────────────────

def get_verdict(score_pct: float) -> tuple:
    """
    Returns (verdict_label, symbol) based on score thresholds.

    Real ATS logic:
      >= SHORTLIST_THRESHOLD  → auto shortlist  (recruiter sees these first)
      >= REVIEW_THRESHOLD     → manual review   (recruiter checks manually)
      <  REVIEW_THRESHOLD     → auto reject     (never shown to recruiter)
    """
    if score_pct >= SHORTLIST_THRESHOLD:
        return "SHORTLISTED",   "[+]"
    elif score_pct >= REVIEW_THRESHOLD:
        return "MANUAL REVIEW", "[?]"
    else:
        return "REJECTED",      "[x]"

# ─────────────────────────────────────────────────────
#  7. CORE RAG SCREEN FUNCTION
#     - embeds JD query
#     - retrieves top 3 resume chunks from ChromaDB
#     - applies threshold → verdict
# ─────────────────────────────────────────────────────

def screen_resume(role: str, company: str, query: str) -> dict:
    """
    Screen a resume against a job description query.
    Returns result dict with score, verdict, and matched chunks.
    """
    # Retrieve top 3 chunks + L2 distance scores
    raw_results = vectorstore.similarity_search_with_score(query, k=3)

    chunks_out = []
    for doc, l2_score in raw_results:
        # Convert L2 distance → similarity percentage
        # Normalized vectors: L2 in [0, 2]  →  score in [0, 100]
        pct = round((1 - l2_score / 2) * 100, 1)
        chunks_out.append({
            "text":  doc.page_content.strip(),
            "score": pct,
        })

    # Best score = top chunk (most relevant section of the resume)
    best_score     = chunks_out[0]["score"] if chunks_out else 0.0
    verdict, symbol = get_verdict(best_score)

    return {
        "role":    role,
        "company": company,
        "score":   best_score,
        "verdict": verdict,
        "symbol":  symbol,
        "chunks":  chunks_out,
    }

# ─────────────────────────────────────────────────────
#  8. RUN ATS SCREENING ACROSS ALL JDs
# ─────────────────────────────────────────────────────

print(SEP2)
print("  ATS RESUME SCREENING")
print(f"  Shortlist >= {SHORTLIST_THRESHOLD}%  |  Review >= {REVIEW_THRESHOLD}%  |  Reject < {REVIEW_THRESHOLD}%")
print(SEP2)

all_results = []

for jd in JOB_DESCRIPTIONS:
    result = screen_resume(jd["role"], jd["company"], jd["query"])
    all_results.append(result)

    bar = "X" * int(result["score"] / 5) + "." * (20 - int(result["score"] / 5))

    print(f"\n  {result['symbol']}  {result['role']:<28} @ {result['company']}")
    print(f"     Score  : [{bar}] {result['score']}%")
    print(f"     Verdict: {result['verdict']}")
    print(f"     Top 3 matched resume chunks:")

    for i, chunk in enumerate(result["chunks"], 1):
        print(f"       [{i}] {chunk['score']}%  -> \"{chunk['text'][:75]} ...\"")

# ─────────────────────────────────────────────────────
#  9. SUMMARY TABLE  — like a real ATS dashboard
# ─────────────────────────────────────────────────────

print(f"\n\n{SEP2}")
print("  SCREENING SUMMARY")
print(SEP2)
print(f"  {'Role':<28} {'Company':<15} {'Score':>7}  {'Verdict'}")
print(f"  {SEP}")

# Sort by score descending
all_results.sort(key=lambda x: x["score"], reverse=True)

for r in all_results:
    print(f"  {r['symbol']}  {r['role']:<26} {r['company']:<15} {r['score']:>5}%   {r['verdict']}")

shortlisted = [r for r in all_results if r["verdict"] == "SHORTLISTED"]
review      = [r for r in all_results if r["verdict"] == "MANUAL REVIEW"]
rejected    = [r for r in all_results if r["verdict"] == "REJECTED"]

print(f"\n  Total screened : {len(all_results)}")
print(f"  [+] Shortlisted : {len(shortlisted)}")
print(f"  [?] Review       : {len(review)}")
print(f"  [x] Rejected     : {len(rejected)}")
print(SEP2)

# ─────────────────────────────────────────────────────
#  10. INTERACTIVE MODE
#      Paste any JD text and screen live against the resume
# ─────────────────────────────────────────────────────

print(f"\n{SEP2}")
print("  INTERACTIVE MODE  (type 'exit' to quit)")
print("  Paste a JD or skill keywords to screen this resume.")
print(SEP2)

while True:
    user_input = input("\n  Your query / JD text: ").strip()

    if user_input.lower() in ("exit", "quit", "q"):
        print("  Bye!")
        break
    if not user_input:
        print("  Please enter a query.")
        continue

    result = screen_resume("Custom Query", "—", user_input)
    bar    = "X" * int(result["score"] / 5) + "." * (20 - int(result["score"] / 5))

    print(f"\n  Score  : [{bar}] {result['score']}%")
    print(f"  Verdict: {result['symbol']} {result['verdict']}")
    print(f"  Top 3 matched chunks:")
    for i, chunk in enumerate(result["chunks"], 1):
        print(f"    [{i}] {chunk['score']}%  -> \"{chunk['text'][:90]} ...\"")