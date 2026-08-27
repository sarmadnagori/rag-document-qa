from fastapi import FastAPI
import psycopg2
import os
from dotenv import load_dotenv
from pydantic import BaseModel
import json
import ollama
import math
from fastapi import UploadFile,HTTPException
from langchain_text_splitters import RecursiveCharacterTextSplitter
import io
import pypdf
import docx
from fastapi.middleware.cors import CORSMiddleware



load_dotenv()
app = FastAPI(
    title="Document Q&A",
    description="A FastAPI service that ingests documents, retrieves relevant passages by meaning, and answers questions grounded in the retrieved text — refusing rather than guessing when nothing relevant is found.",
    version="0.2.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


THRESHOLD = 0.43      # derived from measured score distributions: far-miss max 0.404, answerable min 0.464


class DOCUMENT(BaseModel):
    text: str
    document_name: str


class Hit(BaseModel):
    text: str
    score: float
    document_name: str
    chunk_index: int

class AskResponse(BaseModel):
    query: str
    reply: str
    score: float
    top3: list[Hit]

class SearchResponse(BaseModel):
    query: str
    top3: list[Hit]

DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_HOST = os.getenv("DB_HOST")
DB_PASSWORD = os.getenv("DB_PASSWORD")


def get_conn():
    return psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        host=DB_HOST,
        password=DB_PASSWORD
    )

conn=get_conn()
cursor=conn.cursor()

cursor.execute("""CREATE TABLE IF NOT EXISTS semantic(
    id SERIAL PRIMARY KEY,
    document_name TEXT,
    chunk_index INTEGER,
    text TEXT,
    embedding TEXT
)
""")
conn.commit()
conn.close()
def embed(text):
    return ollama.embeddings(model="nomic-embed-text", prompt=text)["embedding"]

def similarity(v1, v2):
    dot = sum(x * y for x, y in zip(v1, v2))
    mag1 = math.sqrt(sum(x * x for x in v1))
    mag2 = math.sqrt(sum(x * x for x in v2))
    return dot / (mag1 * mag2)


def ask_model(prompt):
    response=ollama.chat(
        model="llama3.2",
        messages=[{"role":"user","content":prompt}],
        options={"temperature": 0}
         

    )
    return response.message.content


def retrieve(q):


     conn=get_conn()
     cursor=conn.cursor()
     

     query_vector=embed(q)
     cursor.execute("SELECT chunk_index,document_name,text,embedding FROM semantic")
     rows=cursor.fetchall()
     results=[]
     for row in rows:
         chunk_vector = json.loads(row[3])
         score=similarity(query_vector,chunk_vector)
         results.append({"text": row[2], "score": score,"document_name":row[1],"chunk_index":row[0]})

         
     
     ranked = sorted(results, key=lambda r: r["score"], reverse=True)
     
     top3 = ranked[:3]
     
         
     conn.close()
     return top3



@app.get(
    "/search",
    response_model=SearchResponse,
    summary="Retrieve the most relevant document chunks",
    description="Embeds the query and returns the three most semantically "
    "similar chunks from the indexed documents, ranked by score, along with "
    "their source document name and chunk index.\n\n"
    "This endpoint performs retrieval only. No model is called, no relevance "
    "threshold is applied, and no answer is generated — chunks are returned "
    "even when nothing is a good match. Use /ask for a generated answer; use "
    "this endpoint to inspect what retrieval is finding.",
)
def get_word(q: str):
    return {"query": q, "top3": retrieve(q)}



def answer(q):
    top3 = retrieve(q)

    if not top3 or top3[0]["score"] < THRESHOLD:
        return {
            "query": q,
            "reply": "I don't know",
            "score": top3[0]["score"] if top3 else 0.0,
            "top3": [],
        }

    context = "\n\n".join([item["text"] for item in top3])

    prompt = f"""Answer the question using only the information provided below.

If the information below does not contain the answer, reply exactly: I don't know.

Answer the question directly. Do not mention the information provided, the context, or the passage.

Give a complete answer in two to four sentences.

Information:
{context}

Question: {q}

Answer:"""

    reply = ask_model(prompt)

    return {
        "query": q,
        "reply": reply,
        "score": top3[0]["score"],
        "top3": top3,
    }


@app.post(
    "/ask",
    response_model=AskResponse,
    summary="Answer a question using the indexed documents",
    description="Embeds the question, retrieves the most similar chunks from the "
    "indexed documents, and generates an answer grounded in that retrieved "
    "text. The response includes the answer along with the source chunks "
    "it was based on.\n\n"
    "If no chunk is similar enough to the question, the endpoint does not "
    "guess — it returns a refusal stating the answer is not present in "
    "the documents. This is expected behaviour, not an error."
)
def ask(q: str):
    return answer(q)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
    separators=["\n\n", "\n", ". ", " ", ""],
)

def extract_pdf(raw):
    reader = pypdf.PdfReader(io.BytesIO(raw))
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)


def extract_docx(raw):
    document = docx.Document(io.BytesIO(raw))
    return "\n\n".join(p.text for p in document.paragraphs)

    
    
@app.post("/documents")
async def get_document(file: UploadFile):
    raw = await file.read()
    name = file.filename

    if name.endswith(".pdf"):
        text = extract_pdf(raw)
    elif name.endswith(".docx"):
        text = extract_docx(raw)
    else:
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            raise HTTPException(
                status_code=400,
                detail="File is not valid UTF-8 text. Supported: .txt, .pdf, .docx",
            )

    chunks = splitter.split_text(text)
    if not chunks:
        raise HTTPException(
            status_code=400,
            detail="No text could be extracted. The file may be a scanned image requiring OCR.",
        )

    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM semantic WHERE document_name=%s", (name,))
    for item, chunk in enumerate(chunks):
        vector = embed(chunk)
        vector_string = json.dumps(vector)
        cursor.execute(
            "INSERT INTO semantic (document_name,chunk_index,text,embedding) VALUES (%s,%s,%s,%s)",
            (name, item, chunk, vector_string),
        )
    conn.commit()
    conn.close()
    return {"chunks_stored": len(chunks)}