RAG API with Multi-Provider LLM + Anti-AI Humanization
Architecture: FastAPI + ChromaDB + Ollama + Multi-LLM failover
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import requests
import random
import re
import os
import time
import logging
import chromadb
from chromadb.config import Settings
import fitz  # pymupdf
from pathlib import Path

# ============ LOGGING ============
Dimitris@DXP6800PRO-2531:~$ grep -iE "sk-[a-zA-Z0-9]|hf_[a-zA-Z0-9]|AIza[a-zA-Z0-9]|gsk_[a-zA-Z0-9]|csk-[a-zA-Z0-9]" ~/rag_system/api/app_current.py
Dimitris@DXP6800PRO-2531:~$ grep -iE "sk-[a-zA-Z0-9]|hf_[a-zA-Z0-9]|AIza[a-zA-Z0-9]|gsk_[a-zA-Z0-9]|csk-[a-zA-Z0-9]" ~/rag_system/api/app_current.py
Dimitris@DXP6800PRO-2531:~$ cat ~/rag_system/api/app_current.py
"""
RAG API with Multi-Provider LLM + Anti-AI Humanization
Architecture: FastAPI + ChromaDB + Ollama + Multi-LLM failover
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import requests
import random
import re
import os
import time
import logging
import chromadb
from chromadb.config import Settings
import fitz  # pymupdf
from pathlib import Path

# ============ LOGGING ============
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============ CONFIG ============
CHROMA_HOST = os.getenv("CHROMA_HOST", "chromadb")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")
BOOKS_PATH = "/books"
COLLECTION_NAME = "herbal_books"

# Load API keys (rotation lists)
GROQ_KEYS = [os.getenv(f"GROQ_API_KEY{'_' + str(i) if i > 1 else ''}") for i in range(1, 7)]
GROQ_KEYS = [k for k in GROQ_KEYS if k]

CEREBRAS_KEYS = [os.getenv(f"CEREBRAS_API_KEY{'_' + str(i) if i > 1 else ''}") for i in range(1, 4)]
CEREBRAS_KEYS = [k for k in CEREBRAS_KEYS if k]

GEMINI_KEY = os.getenv("GEMINI_API_KEY")
MISTRAL_KEY = os.getenv("MISTRAL_API_KEY")

# Rotation counters
groq_idx = 0
cerebras_idx = 0

# ============ FASTAPI ============
app = FastAPI(title="RAG API with Humanization")

# ============ CHROMADB CLIENT ============
chroma_client = None
collection = None

def init_chromadb():
    global chroma_client, collection
    try:
        chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)
        logger.info(f"ChromaDB connected. Collection '{COLLECTION_NAME}' has {collection.count()} documents")
        return True
    except Exception as e:
        logger.error(f"ChromaDB init failed: {e}")
        return False

# ============ EMBEDDINGS via OLLAMA ============
def get_embedding(text: str) -> List[float]:
    """Get embedding from Ollama bge-m3 model."""
    try:
        response = requests.post(
            f"{OLLAMA_HOST}/api/embeddings",
            json={"model": "bge-m3", "prompt": text},
            timeout=60
        )
        return response.json().get("embedding", [])
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        return []

# ============ INDEX PDFs ============
def index_books():
    """Read all PDFs from /books and embed into ChromaDB."""
    books_dir = Path(BOOKS_PATH)
    if not books_dir.exists():
        logger.warning(f"Books directory {BOOKS_PATH} not found")
        return 0

    pdfs = list(books_dir.glob("*.pdf"))
    logger.info(f"Found {len(pdfs)} PDFs to index")

    indexed_count = 0
    for pdf_path in pdfs:
        try:
            doc = fitz.open(pdf_path)
            book_name = pdf_path.stem

            for page_num, page in enumerate(doc):
                text = page.get_text()
                if len(text.strip()) < 100:
                    continue

                # Chunk by paragraph (~500 char chunks)
                chunks = [text[i:i+500] for i in range(0, len(text), 500)]

                for chunk_idx, chunk in enumerate(chunks):
                    if len(chunk.strip()) < 50:
                        continue
                    doc_id = f"{book_name}_p{page_num}_c{chunk_idx}"
                    embedding = get_embedding(chunk)
                    if embedding:
                        try:
                            collection.add(
                                ids=[doc_id],
                                embeddings=[embedding],
                                documents=[chunk],
                                metadatas=[{"book": book_name, "page": page_num}]
                            )
                            indexed_count += 1
                        except Exception:
                            pass  # already indexed
            doc.close()
            logger.info(f"Indexed: {book_name}")
        except Exception as e:
            logger.error(f"Failed indexing {pdf_path}: {e}")

    return indexed_count

# ============ RAG RETRIEVAL ============
def retrieve_context(query: str, top_k: int = 5) -> str:
    """Retrieve relevant chunks from ChromaDB."""
    if not collection:
        return ""
    try:
        query_embedding = get_embedding(query)
        if not query_embedding:
            return ""

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        if results and results.get("documents"):
            chunks = results["documents"][0]
            return "\n\n---\n\n".join(chunks)
        return ""
    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        return ""

# ============ MULTI-PROVIDER LLM ROUTER ============

def call_groq(prompt: str, temperature: float = 1.0, max_tokens: int = 4000) -> Optional[str]:
    """Try all Groq keys in rotation."""
    global groq_idx
    for attempt in range(len(GROQ_KEYS)):
        key = GROQ_KEYS[groq_idx]
        groq_idx = (groq_idx + 1) % len(GROQ_KEYS)
        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json={
                    "model": "llama-3.3-70b-versatile",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "top_p": 0.95,
                },
                timeout=120
            )
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            logger.warning(f"Groq key {groq_idx} failed: {response.status_code}")
        except Exception as e:
            logger.warning(f"Groq error: {e}")
    return None

def call_cerebras(prompt: str, temperature: float = 1.0, max_tokens: int = 4000) -> Optional[str]:
    """Try all Cerebras keys in rotation."""
    global cerebras_idx
    for attempt in range(len(CEREBRAS_KEYS)):
        key = CEREBRAS_KEYS[cerebras_idx]
        cerebras_idx = (cerebras_idx + 1) % len(CEREBRAS_KEYS)
        try:
            response = requests.post(
                "https://api.cerebras.ai/v1/chat/completions",
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json={
                    "model": "llama-3.3-70b",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
                timeout=120
            )
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            logger.warning(f"Cerebras error: {e}")
    return None

def call_gemini(prompt: str, temperature: float = 1.0) -> Optional[str]:
    """Call Gemini."""
    if not GEMINI_KEY:
        return None
    try:
        response = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_KEY}",
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": temperature, "topP": 0.95, "maxOutputTokens": 4000}
            },
            timeout=120
        )
        if response.status_code == 200:
            return response.json()["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        logger.warning(f"Gemini error: {e}")
    return None

def call_mistral(prompt: str, temperature: float = 1.0) -> Optional[str]:
    """Call Mistral."""
    if not MISTRAL_KEY:
        return None
    try:
        response = requests.post(
            "https://api.mistral.ai/v1/chat/completions",
            headers={"Authorization": f"Bearer {MISTRAL_KEY}", "Content-Type": "application/json"},
            json={
                "model": "mistral-large-latest",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": 4000,
            },
            timeout=120
        )
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        logger.warning(f"Mistral error: {e}")
    return None

def call_llm(prompt: str, temperature: float = 1.0, max_tokens: int = 4000) -> str:
    """Smart router: try providers in order until one works."""
    # Order: Groq → Cerebras → Gemini → Mistral
    for func in [call_groq, call_cerebras, call_gemini, call_mistral]:
        result = func(prompt, temperature, max_tokens) if func.__name__ in ['call_groq', 'call_cerebras'] else func(prompt, temperature)
        if result:
            logger.info(f"LLM success via {func.__name__}")
            return result
    raise HTTPException(status_code=503, detail="All LLM providers failed")

# ============ HUMANIZER ============

BANNED_WORDS = {
    "delve": "look at", "tapestry": "mix", "navigate": "get through",
    "landscape": "scene", "realm": "area", "crucial": "important",
    "moreover": "also", "furthermore": "plus", "comprehensive": "full",
    "leverage": "use", "robust": "strong", "seamless": "smooth",
    "dive into": "look at", "harness": "use", "elevate": "lift",
    "empower": "help", "unlock": "open up", "unleash": "let loose",
    "in conclusion": "so basically", "it's important to note": "worth noting",
    "in today's fast-paced world": "these days",
    "when it comes to": "with", "in the ever-evolving": "in the changing",
}

def remove_banned_words(text: str) -> str:
    for banned, replacement in BANNED_WORDS.items():
        text = re.compile(re.escape(banned), re.IGNORECASE).sub(replacement, text)
    return text

def vary_sentence_length(text: str) -> str:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    result = []
    for i, sent in enumerate(sentences):
        words = sent.split()
        if i % 4 == 0 and len(words) > 20:
            mid = len(words) // 2
            result.append(' '.join(words[:mid]) + '.')
            result.append(' '.join(words[mid:]))
        elif i % 7 == 0 and len(words) > 8:
            result.append(sent)
            result.append(random.choice(["True story.", "Seriously.", "No joke.", "Trust me.", "It happens."]))
        else:
            result.append(sent)
    return ' '.join(result)

def add_human_starters(text: str) -> str:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    starters = ["And ", "But ", "So ", "Honestly, ", "Look, ", "Now, "]
    for i in range(len(sentences)):
        if i > 0 and random.random() < 0.12 and len(sentences[i]) > 5:
            sentences[i] = random.choice(starters) + sentences[i][0].lower() + sentences[i][1:]
    return ' '.join(sentences)

def inconsistent_contractions(text: str) -> str:
    pairs = [("do not", "don't"), ("does not", "doesn't"), ("it is", "it's"),
             ("you are", "you're"), ("cannot", "can't"), ("will not", "won't")]
    for full, contracted in pairs:
        def replacer(match):
            return contracted if random.random() < 0.5 else match.group(0)
        text = re.sub(re.escape(full), replacer, text, flags=re.IGNORECASE)
    return text

def add_hedges(text: str) -> str:
    hedges = ["I think ", "Honestly, ", "In my experience, ", "Probably ", "I'd say ", "From what I've seen, "]
    paragraphs = text.split('\n\n')
    for i, para in enumerate(paragraphs):
        if len(para) > 200 and random.random() < 0.4:
            sentences = para.split('. ')
            if len(sentences) > 2:
                pos = random.randint(1, len(sentences) - 1)
                sentences[pos] = random.choice(hedges) + sentences[pos][0].lower() + sentences[pos][1:]
                paragraphs[i] = '. '.join(sentences)
    return '\n\n'.join(paragraphs)

def add_rhetorical_questions(text: str, per_300_words: int = 1) -> str:
    questions = ["Sound familiar?", "Make sense?", "Ever noticed that?", "Why does this matter?", "Surprised?"]
    total_words = len(text.split())
    insert_count = (total_words // 300) * per_300_words
    paragraphs = text.split('\n\n')
    for _ in range(insert_count):
        if paragraphs:
            idx = random.randint(0, len(paragraphs) - 1)
            paragraphs[idx] += " " + random.choice(questions)
    return '\n\n'.join(paragraphs)

def back_translate_chunk(text: str, chain: List[str]) -> str:
    """Back-translate via free MyMemory API."""
    current = text
    for i in range(len(chain) - 1):
        try:
            response = requests.get(
                "https://api.mymemory.translated.net/get",
                params={"q": current[:500], "langpair": f"{chain[i]}|{chain[i+1]}"},
                timeout=30
            )
            if response.status_code == 200:
                translated = response.json().get("responseData", {}).get("translatedText", current)
                if translated and len(translated) > 10:
                    current = translated
            time.sleep(0.5)  # rate limit
        except Exception as e:
            logger.warning(f"Translation failed: {e}")
    return current

def back_translate(text: str, chain: List[str]) -> str:
    paragraphs = text.split('\n\n')
    result = []
    for para in paragraphs:
        if len(para) > 50 and len(para) < 500:
            result.append(back_translate_chunk(para, chain))
        else:
            result.append(para)
    return '\n\n'.join(result)

def full_humanize(text: str, settings: dict) -> str:
    """Apply all humanization steps."""
    text = remove_banned_words(text)
    if settings.get("vary_sentence_length", True):
        text = vary_sentence_length(text)
    text = add_human_starters(text)
    if settings.get("inconsistent_contractions", True):
        text = inconsistent_contractions(text)
    if settings.get("add_hedges", True):
        text = add_hedges(text)
    text = add_rhetorical_questions(text, settings.get("rhetorical_questions_per_300_words", 1))

    chain = settings.get("back_translation_chain", [])
    if chain and len(chain) > 1:
        text = back_translate(text, chain)

    text = remove_banned_words(text)
    return text

# ============ ENDPOINTS ============

class ArticleRequest(BaseModel):
    keyword: str
    outline: Optional[dict] = None
    language: str = "en"
    word_count: int = 1500
    humanize: bool = True
    extreme_mode: bool = False
    anti_ai_settings: Optional[dict] = None
    seo_settings: Optional[dict] = None

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

@app.on_event("startup")
async def startup():
    init_chromadb()
    if collection and collection.count() == 0:
        logger.info("Empty collection — starting auto-indexing of books...")
        count = index_books()
        logger.info(f"Indexed {count} chunks")

@app.get("/")
async def root():
    return {"status": "ok", "service": "RAG API with Humanization"}

@app.get("/health")
async def health():
    doc_count = collection.count() if collection else 0
    return {
        "status": "healthy",
        "chromadb": "connected" if collection else "disconnected",
        "documents": doc_count,
        "providers": {
            "groq_keys": len(GROQ_KEYS),
            "cerebras_keys": len(CEREBRAS_KEYS),
            "gemini": bool(GEMINI_KEY),
            "mistral": bool(MISTRAL_KEY),
        }
    }

@app.post("/reindex")
async def reindex():
    """Manually trigger re-indexing of books."""
    count = index_books()
    return {"indexed_chunks": count}

@app.post("/search-books")
async def search_books(req: SearchRequest):
    context = retrieve_context(req.query, req.top_k)
    return {"context": context, "query": req.query}

@app.post("/generate-article")
async def generate_article(req: ArticleRequest):
    logger.info(f"Generating article for: {req.keyword}")

    # 1. RAG retrieval
    rag_context = retrieve_context(req.keyword, top_k=5)

    # 2. Build prompt
    settings = req.anti_ai_settings or {}
    banned = ", ".join(list(BANNED_WORDS.keys()))
    outline_text = str(req.outline) if req.outline else "Create your own structure with 6-8 sections."

    prompt = f"""Write a {req.word_count}-word SEO article about: {req.keyword}

Context from books (use this as source material):
{rag_context[:3000]}

Outline to follow:
{outline_text}

CRITICAL STYLE RULES (sound human, not AI):
- Mix sentence length AGGRESSIVELY: some 3 words, some 25+ words
- Use sentence fragments. Like this. On purpose.
- Add hedges: "I think", "probably", "honestly", "kinda", "I'd say"
- Start 10-15% of sentences with And, But, So, Honestly, Look
- Use contractions INCONSISTENTLY (don't here, do not there)
- Insert 1 rhetorical question per 300 words
- Include 2 personal opinions per section
- Write like a tired blogger at 11pm

BANNED WORDS (NEVER use): {banned}

SEO RULES:
- Use "{req.keyword}" 6-10 times naturally throughout
- Include keyword in first 100 words
- Bold 3-5 key phrases per section using **markdown**
- End with FAQ section (3-5 questions)

Write the full article in Markdown now. No intro about what you're doing — just the article."""

    # 3. Generate
    article = call_llm(
        prompt,
        temperature=settings.get("temperature", 1.0),
        max_tokens=4000
    )

    # 4. Humanize
    if req.humanize:
        article = full_humanize(article, settings)

    # 5. Extreme mode = second pass
    if req.extreme_mode:
        article = full_humanize(article, settings)

    return {
        "article": article,
        "keyword": req.keyword,
        "word_count": len(article.split()),
        "humanized": req.humanize,
        "extreme_mode": req.extreme_mode,
        "rag_used": bool(rag_context),
    }
Dimitris@DXP6800PRO-2531:~$
