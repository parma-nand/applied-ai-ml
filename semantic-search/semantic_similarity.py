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

RESUME_TEXT = """Parma Nand
+91-98765-43210 | Pune, Maharashtra, India | parmanand@email.com
linkedin.com/in/parmanand | github.com/parma-nand
Professional Summary
Results-driven AI/ML Engineer with 4 years of experience designing and deploying intelligent systems, agentic AI pipelines,
and production-grade machine learning solutions. Proficient in Python, LangChain, LangGraph, and Apache Airflow.
Proven track record building end-to-end AI applications including resume parsing engines, voice-based chat systems, and
intelligent recommendation engines. Experienced in Generative AI, LLM orchestration, and MLOps workflows.
Technical Skills
Languages: Python, SQL, Bash
AI / ML Frameworks: LangChain, LangGraph, Hugging Face Transformers, scikit-learn, TensorFlow, Keras, PyTorch
Agentic AI: Agentic AI Pipelines, ReAct Agents, Tool-Calling Agents, Multi-Agent Systems, LangGraph Workflows
Orchestration & MLOps: Apache Airflow, Docker, Docker Compose, FastAPI, MLflow
NLP & Retrieval: spaCy, sentence-transformers, FAISS, Vector Databases (ChromaDB, Pinecone), RAG
Data & Cloud: PySpark, Pandas, NumPy, PostgreSQL, AWS (S3, Lambda), GCP
Tools: Git, GitHub, VS Code, Jupyter, Postman, Streamlit
Work Experience
Associate Consultant– AI/ML Engineering
Capgemini
July 2022– Present
Pune, Maharashtra, India
• Designed and deployed agentic AI pipelines using LangGraph and LangChain, automating multi-step reasoning
workflows that reduced manual processing time by 40%.
• Built and maintained Apache Airflow DAGs to orchestrate end-to-end ML training, evaluation, and model
deployment pipelines across cloud environments.
• Developed a production-ready resume parsing system using spaCy NER, pdfplumber, and sentence-transformers,
achieving 92% field extraction accuracy across diverse resume formats.
• Engineered a voice-based AI chat application integrating speech-to-text, LLM inference (via LangChain), and
text-to-speech, deployed as a containerized FastAPI microservice.
• Architected an intelligent recommendation engine using cosine similarity with sentence-transformers embeddings,
surfacing ranked job matches for over 10,000 candidate profiles.
• Implemented RAG (Retrieval-Augmented Generation) pipelines using FAISS and ChromaDB vector stores to power
context-aware LLM responses for enterprise knowledge bases.
• Collaborated with cross-functional teams to deliver AI-powered features in Agile sprints, contributing to
client-facing demos and technical documentation.
• Containerized AI services using Docker and Docker Compose, ensuring reproducible deployments across dev,
staging, and production environments.
Projects
Resume Parsing & Job Matcher | Python, FastAPI, spaCy, sentence-transformers, Streamlit, Docker
• Built a full-stack resume analysis platform with a FastAPI backend and Streamlit frontend, supporting
PDF/DOCX uploads up to 14MB with PyMuPDF/pdfplumber fallback extraction.
• Implemented spaCy NER for extracting companies, skills, education, and experience dates; integrated
BART-large-CNN (Hugging Face) for resume summarization.
2024
• Developed a cosine-similarity scoring module using sentence-transformers to rank job descriptions against parsed
resumes, improving match relevance by 35%.
• Containerized the entire stack with Docker Compose; integrated Adzuna and JSearch APIs for real-time job
recommendations.
Voice Chat AI Assistant | Python, LangChain, FastAPI, Whisper, TTS, Docker
2023
• Developed a voice-driven conversational AI system using OpenAI Whisper for speech-to-text and a TTS engine for semantic-search\semantic_similarity.py
audio responses.
• Orchestrated multi-turn dialogue management with LangChain’s memory modules, maintaining conversation
context across sessions.
• Deployed as a REST API with FastAPI, containerized with Docker, supporting concurrent voice sessions with
sub-2-second response latency.
Agentic AI Recommendation Engine | Python, LangGraph, FAISS, ChromaDB, FastAPI
• Designed a multi-agent recommendation workflow using LangGraph with specialized agents for query
understanding, retrieval, ranking, and response generation.
2024
• Integrated FAISS vector search over dense embeddings to retrieve semantically relevant items from a corpus of
50,000+ records with under 100ms latency.
• Exposed the pipeline via FastAPI with Pydantic-validated request/response schemas and integrated logging for
model observability.
Education
National Institute of Technology (NIT) Jalandhar
Bachelor of Technology (B.Tech)
Certifications
Jalandhar, Punjab
August 2018– May 2022
DeepLearning.AI– LangChain for LLM Application Development
Coursera– Machine Learning Specialization (Andrew Ng)
Databricks– Apache Spark Fundamental

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