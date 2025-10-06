from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status, Request, Body, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
import gensim
import numpy as np
from gensim.models import Word2Vec, FastText
import tempfile
import re
import nltk
import logging
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.tokenize import RegexpTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import uvicorn
from pathlib import Path
import json
import time
from typing import List, Optional, Dict, Any, Tuple
import asyncio
from contextlib import asynccontextmanager
from rank_bm25 import BM25Okapi
import openai
from dataclasses import dataclass
import math
from collections import defaultdict

# Load environment variables
load_dotenv()

# Konfigurasi logging yang lebih baik
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Global variables untuk model cache
model_cache = {}
bm25_cache = {}

# OpenAI Configuration
openai.api_key = os.getenv("OPENAI_API_KEY")

# Default Prompt for generate_answer_with_gpt
DEFAULT_SYSTEM_PROMPT ="""AI assistant is a professional and polite customer service work at PT. Omni Hottilier representative. 
The traits of AI include expert knowledge, helpfulness, cleverness, and articulateness. 
AI assistant provides clear, concise, and friendly responses without repeating unnecessary information or phrases such as "Berdasarkan informasi yang diberikan sebelumnya.", "dalam konteks yang diberikan.", "dalam konteks yang tersedia.".
AI is a well-behaved and well-mannered individual. 
AI is always friendly, kind, and inspiring, and he is eager to provide vivid and thoughtful responses to the user. 
AI has the sum of all knowledge in their brain, and is able to accurately answer nearly any question about any topic in conversation. 
AI assistant make answer using Indonesian Language. 
AI assistant avoids sounding repetitive and ensures responses sound natural and tailored to each question. 
If the context does not provide the answer to question, the AI assistant will say, "Mohon Maaf, tapi saya tidak dapat menjawab pertanyaan tersebut saat ini.".
AI assistant will take into account any CONTEXT BLOCK that is provided in a conversation. 
AI assistant will not apologize for previous responses, but instead will indicated new information was gained. 
AI assistant will not invent anything that is not drawn directly from the context."""

@dataclass
class QueryComplexity:
    """Analisis kompleksitas query"""
    word_count: int
    unique_words: int
    question_words: int
    complexity_score: float
    query_type: str  # simple, medium, complex

@dataclass
class SearchResult:
    """Hasil pencarian dengan metadata lengkap"""
    text: str
    fasttext_similarity: float
    bm25_score: float
    context_score: float
    weighted_score: float
    doc_index: int
    context_range: str

@dataclass
class MMRResult:
    """Hasil setelah MMR reranking"""
    text: str
    final_score: float
    diversity_penalty: float
    original_rank: int
    doc_index: int

@dataclass
class RAGASMetrics:
    """Metrik evaluasi RAGAS"""
    context_relevance: float
    faithfulness: float
    answer_relevance: float
    overall_score: float

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup dan shutdown events"""
    # Startup
    logger.info("Starting FastAPI application...")
    
    # Download NLTK data jika belum ada
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
    except LookupError:
        logger.info("Downloading NLTK data...")
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
    
    # Buat direktori yang diperlukan
    Path("model").mkdir(exist_ok=True)
    Path("storage").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    
    yield
    
    # Shutdown
    logger.info("Shutting down FastAPI application...")
    model_cache.clear()

app = FastAPI(
    title="Document Search API",
    description="API untuk pencarian dokumen menggunakan Word2Vec dan FastText",
    version="1.0.0",
    lifespan=lifespan
)

# # Setup CORS dengan konfigurasi yang lebih aman
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=os.getenv("ALLOWED_ORIGINS", "http://localhost:9999").split(","),
#     allow_credentials=True,
#     allow_methods=["GET", "POST"],
#     allow_headers=["*"],
# )

# Konfigurasi
EMBEDDING_DIM = 100  # Dikurangi untuk performa yang lebih baik
MAX_FILE_SIZE = 100 * 1024 * 1024  # 10MB
MIN_CHUNK_LENGTH = 300  # Minimum panjang chunk
MAX_CHUNK_SIZE = 1000  # Maximum chunk size

# Inisialisasi stemmer dan tokenizer
factory = StemmerFactory()
stemmer = factory.create_stemmer()
tokenizer = RegexpTokenizer(r'\w+')

def get_stopwords() -> set:
    """Dapatkan stopwords bahasa Indonesia yang diperluas"""
    try:
        stop_words = set(stopwords.words('indonesian'))
    except Exception:
        stop_words = set()
    
    custom_stopwords = {
        'yg', 'dg', 'dgn', 'ny', 'sih', 'nya', 'kalo', 'deh', 'mah',
        'lah', 'dll', 'tsb', 'dr', 'pd', 'utk', 'sd', 'dpt', 'dlm',
        'thn', 'tgl', 'jd', 'tkr', 'org', 'sbg', 'bs', 'kpd', 'yng',
        'dgn', 'krn', 'bhw', 'shg', 'dri', 'ke', 'di', 'dari', 'untuk'
    }
    
    return stop_words.union(custom_stopwords)

def preprocess_text(text: str) -> List[str]:
    """Preprocessing teks yang dioptimalkan untuk bahasa Indonesia"""
    try:
        if not text or not isinstance(text, str):
            return []
        
        # Normalisasi teks
        text = text.lower().strip()
        
        # Hapus karakter khusus dan angka
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        
        # Tokenization
        tokens = tokenizer.tokenize(text)
        
        if not tokens:
            return []
        
        # Stemming dan filtering
        stop_words = get_stopwords()
        
        processed_tokens = []
        for token in tokens:
            if (len(token) > 2 and 
                token not in stop_words and 
                not token.isdigit() and
                token.isalpha()):  # Hanya huruf
                
                stemmed = stemmer.stem(token)
                if len(stemmed) > 2:
                    processed_tokens.append(stemmed)
        
        return processed_tokens
    
    except Exception as e:
        logger.error(f"Preprocessing error for text: {str(e)}")
        return []

def get_model_cache_key(userId: str, chatbotId: str, pdfTitle: str, modelType: str) -> str:
    """Generate cache key untuk model"""
    return f"{userId}_{chatbotId}_{pdfTitle}_{modelType}"

def load_model_cached(model_path: Path, modelType: str, cache_key: str):
    """Load model dengan caching"""
    if cache_key in model_cache:
        logger.info(f"Using cached model: {cache_key}")
        return model_cache[cache_key]
    
    try:
        if modelType == 'word2vec':
            model = Word2Vec.load(str(model_path))
        else:
            model = FastText.load(str(model_path))
        
        # Cache model (dengan batasan ukuran cache)
        if len(model_cache) < 10:  # Maksimal 10 model di cache
            model_cache[cache_key] = model
            logger.info(f"Model cached: {cache_key}")
        
        return model
    
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise HTTPException(404, f"Model tidak ditemukan: {str(e)}")

def analyze_query_complexity(query: str) -> QueryComplexity:
    """Analisis kompleksitas query untuk menentukan weight strategy"""
    tokens = preprocess_text(query)
    raw_tokens = query.lower().split()
    
    question_words = ['apa', 'siapa', 'kapan', 'dimana', 'mengapa', 'bagaimana', 'berapa']
    question_count = sum(1 for word in raw_tokens if word in question_words)
    
    word_count = len(raw_tokens)
    unique_words = len(set(raw_tokens))
    
    # Hitung complexity score
    complexity_score = (
        (word_count / 10) * 0.3 +  # Panjang query
        (unique_words / word_count if word_count > 0 else 0) * 0.4 +  # Keragaman kata
        (question_count > 0) * 0.3  # Keberadaan question words
    )
    
    # Tentukan tipe query
    if complexity_score < 0.3:
        query_type = "simple"
    elif complexity_score < 0.7:
        query_type = "medium"
    else:
        query_type = "complex"
    
    return QueryComplexity(
        word_count=word_count,
        unique_words=unique_words,
        question_words=question_count,
        complexity_score=complexity_score,
        query_type=query_type
    )

def get_weight_strategy(complexity: QueryComplexity) -> Dict[str, float]:
    """Tentukan weight berdasarkan kompleksitas query"""
    if complexity.query_type == "simple":
        # Query sederhana, prioritas keyword matching
        return {
            "fasttext": 0.3,
            "bm25": 0.5,
            "context": 0.2
        }
    elif complexity.query_type == "medium":
        # Query menengah, balanced approach
        return {
            "fasttext": 0.4,
            "bm25": 0.4,
            "context": 0.2
        }
    else:
        # Query kompleks, prioritas semantic understanding
        return {
            "fasttext": 0.5,
            "bm25": 0.3,
            "context": 0.2
        }

def calculate_context_score(doc_text: str, query: str) -> float:
    """Hitung context score berdasarkan berbagai faktor"""
    # Length penalty (dokumen terlalu pendek atau panjang)
    length_penalty = 1.0
    if len(doc_text) < 50:
        length_penalty = 0.5
    elif len(doc_text) > 2000:
        length_penalty = 0.8
    
    # Sentence completeness (apakah ada kalimat lengkap)
    sentence_markers = ['.', '!', '?']
    has_complete_sentence = any(marker in doc_text for marker in sentence_markers)
    completeness_score = 1.0 if has_complete_sentence else 0.7
    
    # Query term coverage
    query_terms = set(preprocess_text(query))
    doc_terms = set(preprocess_text(doc_text))
    coverage = len(query_terms.intersection(doc_terms)) / len(query_terms) if query_terms else 0
    
    return length_penalty * completeness_score * coverage

def create_bm25_index(documents: List[str], cache_key: str) -> BM25Okapi:
    """Buat atau ambil BM25 index dari cache"""
    if cache_key in bm25_cache:
        return bm25_cache[cache_key]
    
    # Preprocess documents untuk BM25
    processed_docs = [preprocess_text(doc) for doc in documents]
    processed_docs = [doc for doc in processed_docs if doc]  # Filter empty
    
    if not processed_docs:
        raise ValueError("Tidak ada dokumen valid untuk BM25 indexing")
    
    bm25 = BM25Okapi(processed_docs)
    
    # Cache dengan batasan ukuran
    if len(bm25_cache) < 10:
        bm25_cache[cache_key] = bm25
    
    return bm25

def mmr_reranking(
    results: List[SearchResult], 
    lambda_param: float = 0.7, 
    top_k: int = 5
) -> List[MMRResult]:
    """Maximal Marginal Relevance reranking untuk diversity"""
    if not results:
        return []
    
    # Sort berdasarkan weighted score
    sorted_results = sorted(results, key=lambda x: x.weighted_score, reverse=True)
    
    selected = []
    remaining = sorted_results.copy()
    
    # Pilih yang pertama (relevance tertinggi)
    if remaining:
        first = remaining.pop(0)
        selected.append(MMRResult(
            text=first.text,
            final_score=first.weighted_score,
            diversity_penalty=0.0,
            original_rank=0,
            doc_index=first.doc_index
        ))
    
    # Iteratively select berdasarkan MMR score
    while remaining and len(selected) < top_k:
        best_mmr_score = -float('inf')
        best_idx = -1
        
        for i, candidate in enumerate(remaining):
            # Hitung similarity dengan yang sudah dipilih
            max_similarity = 0.0
            candidate_tokens = set(preprocess_text(candidate.text))
            
            for selected_result in selected:
                selected_tokens = set(preprocess_text(selected_result.text))
                
                if candidate_tokens and selected_tokens:
                    intersection = len(candidate_tokens.intersection(selected_tokens))
                    union = len(candidate_tokens.union(selected_tokens))
                    jaccard_sim = intersection / union if union > 0 else 0
                    max_similarity = max(max_similarity, jaccard_sim)
            
            # MMR score = λ * relevance - (1-λ) * max_similarity
            mmr_score = (lambda_param * candidate.weighted_score - 
                        (1 - lambda_param) * max_similarity)
            
            if mmr_score > best_mmr_score:
                best_mmr_score = mmr_score
                best_idx = i
        
        if best_idx >= 0:
            selected_candidate = remaining.pop(best_idx)
            selected.append(MMRResult(
                text=selected_candidate.text,
                final_score=best_mmr_score,
                diversity_penalty=(1 - lambda_param) * max_similarity,
                original_rank=len(selected),
                doc_index=selected_candidate.doc_index
            ))
    
    return selected


def calculate_embedding(tokens: List[str], model) -> Optional[np.ndarray]:
    """Hitung embedding rata-rata untuk tokens"""
    if not tokens:
        return None
    
    valid_embeddings = []
    for token in tokens:
        if token in model.wv:
            valid_embeddings.append(model.wv[token])
    
    if not valid_embeddings:
        return None
    
    return np.mean(valid_embeddings, axis=0)

async def generate_answer_with_gpt(
    query: str, 
    contexts: List[MMRResult], 
    model: str = "gpt-3.5-turbo",
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    max_tokens: int = 500,
    temperature: float = 0.7
) -> Dict[str, Any]:
    """Generate jawaban menggunakan OpenAI GPT dengan format prompt baru"""
    if not openai.api_key:
        raise HTTPException(500, "OpenAI API key tidak dikonfigurasi")
    
    # Siapkan context text
    context_text = "\n".join([ctx.text for ctx in contexts])
    
    user_prompt = f"""Pertanyaan: {query}

START CONTEXT BLOCK
{context_text}
END OF CONTEXT BLOCK

Berdasarkan konteks di atas, berikan jawaban yang jelas dan langsung dalam bahasa Indonesia:"""
    
    try:
        response = await openai.ChatCompletion.acreate(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=30
        )
        
        answer = response.choices[0].message.content
        usage = response.usage
        
        # Bersihkan jawaban dari frasa yang tidak diinginkan
        unwanted_phrases = [
            "Berdasarkan konteks di atas",
            "Berdasarkan informasi yang diberikan",
            "dalam konteks yang tersedia",
            "dalam CONTEXT BLOCK"
        ]
        
        for phrase in unwanted_phrases:
            answer = answer.replace(phrase, "")
        
        return {
            "answer": answer.strip(),
            "model_used": model,
            "tokens_used": usage.total_tokens,
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens
        }
        
    except Exception as e:
        logger.error(f"Error generating answer with GPT: {str(e)}")
        raise HTTPException(500, f"Error generating answer: {str(e)}")

def calculate_ragas_metrics(
    query: str, 
    contexts: List[str], 
    answer: str
) -> RAGASMetrics:
    """Implementasi sederhana metrik RAGAS"""
    
    def context_relevance(query: str, contexts: List[str]) -> float:
        """Relevansi konteks terhadap query"""
        if not contexts:
            return 0.0
        
        query_terms = set(preprocess_text(query))
        if not query_terms:
            return 0.0
        
        relevance_scores = []
        for context in contexts:
            context_terms = set(preprocess_text(context))
            if context_terms:
                intersection = len(query_terms.intersection(context_terms))
                relevance = intersection / len(query_terms)
                relevance_scores.append(relevance)
        
        return sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0
    
    def faithfulness(contexts: List[str], answer: str) -> float:
        """Faithfulness answer terhadap context"""
        if not contexts or not answer:
            return 0.0
        
        answer_terms = set(preprocess_text(answer))
        if not answer_terms:
            return 0.0
        
        context_terms = set()
        for context in contexts:
            context_terms.update(preprocess_text(context))
        
        if not context_terms:
            return 0.0
        
        # Hitung berapa banyak terms dalam answer yang ada di context
        supported_terms = len(answer_terms.intersection(context_terms))
        return supported_terms / len(answer_terms)
    
    def answer_relevance(query: str, answer: str) -> float:
        """Relevansi answer terhadap query"""
        if not query or not answer:
            return 0.0
        
        query_terms = set(preprocess_text(query))
        answer_terms = set(preprocess_text(answer))
        
        if not query_terms or not answer_terms:
            return 0.0
        
        intersection = len(query_terms.intersection(answer_terms))
        return intersection / len(query_terms)
    
    # Hitung semua metrik
    ctx_relevance = context_relevance(query, contexts)
    faith_score = faithfulness(contexts, answer)
    ans_relevance = answer_relevance(query, answer)
    
    # Overall score (weighted average)
    overall = (ctx_relevance * 0.4 + faith_score * 0.4 + ans_relevance * 0.2)
    
    return RAGASMetrics(
        context_relevance=ctx_relevance,
        faithfulness=faith_score,
        answer_relevance=ans_relevance,
        overall_score=overall
    )

def hybrid_search(
    query: str,
    fasttext_model,
    bm25_index: BM25Okapi,
    documents: List[str],
    weights: Dict[str, float],
    top_k: int = 10
) -> List[SearchResult]:
    """Hybrid search combining FastText, BM25, and Context scoring"""
    
    if not documents:
        return []
    
    results = []
    processed_query = preprocess_text(query)
    
    # 1. FastText similarity scores
    fasttext_scores = {}
    if processed_query and fasttext_model:
        query_embedding = calculate_embedding(processed_query, fasttext_model)
        if query_embedding is not None:
            for i, doc in enumerate(documents):
                doc_tokens = preprocess_text(doc)
                doc_embedding = calculate_embedding(doc_tokens, fasttext_model)
                if doc_embedding is not None:
                    similarity = cosine_similarity([query_embedding], [doc_embedding])[0][0]
                    fasttext_scores[i] = float(similarity)
    
    # 2. BM25 scores
    bm25_scores = {}
    if processed_query and bm25_index:
        scores = bm25_index.get_scores(processed_query)
        # Normalize BM25 scores to 0-1 range
        if len(scores) > 0 and max(scores) > 0:
            max_score = max(scores)
            bm25_scores = {i: score/max_score for i, score in enumerate(scores)}
    
    # 3. Context scores
    context_scores = {}
    for i, doc in enumerate(documents):
        context_scores[i] = calculate_context_score(doc, query)
    
    # 4. Combine scores
    for i, doc in enumerate(documents):
        fasttext_sim = fasttext_scores.get(i, 0.0)
        bm25_score = bm25_scores.get(i, 0.0)
        context_score = context_scores.get(i, 0.0)
        
        # Weighted combination
        weighted_score = (
            weights["fasttext"] * fasttext_sim +
            weights["bm25"] * bm25_score +
            weights["context"] * context_score
        )
        
        if weighted_score > 0:  # Only include results with positive scores
            # Create context with surrounding documents
            start = max(0, i - 1)
            end = min(len(documents), i + 2)
            context_text = ' '.join(documents[start:end]).strip()
            
            results.append(SearchResult(
                text=context_text,
                fasttext_similarity=fasttext_sim,
                bm25_score=bm25_score,
                context_score=context_score,
                weighted_score=weighted_score,
                doc_index=i,
                context_range=f"{start}-{end-1}"
            ))
    
    # Sort by weighted score and return top-k
    results.sort(key=lambda x: x.weighted_score, reverse=True)
    return results[:top_k]
    """Hitung embedding rata-rata untuk tokens"""
    if not tokens:
        return None
    
    valid_embeddings = []
    for token in tokens:
        if token in model.wv:
            valid_embeddings.append(model.wv[token])
    
    if not valid_embeddings:
        return None
    
    return np.mean(valid_embeddings, axis=0)

def find_context(query: str, embedding_model, docs: List[str], 
                top_k_max: int = 5, similarity_threshold: float = 0.4) -> List[dict]:
    """Pencarian konteks yang dioptimalkan dengan informasi similarity"""
    
    if not query or not docs:
        return []
    
    # Preprocess query
    processed_query = preprocess_text(query)
    if not processed_query:
        logger.warning("Query preprocessing menghasilkan token kosong")
        return []
    
    # Hitung embedding query
    query_embedding = calculate_embedding(processed_query, embedding_model)
    if query_embedding is None:
        logger.warning("Tidak dapat menghitung embedding untuk query")
        return []
    
    # Hitung embedding untuk semua dokumen
    doc_embeddings = []
    valid_doc_indices = []
    
    for i, doc in enumerate(docs):
        processed_doc = preprocess_text(doc)
        doc_embedding = calculate_embedding(processed_doc, embedding_model)
        
        if doc_embedding is not None:
            doc_embeddings.append(doc_embedding)
            valid_doc_indices.append(i)
    
    if not doc_embeddings:
        logger.warning("Tidak ada dokumen yang valid untuk pencarian")
        return []
    
    # Hitung similarity
    similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
    
    # Filter berdasarkan threshold
    relevant_indices = np.where(similarities >= similarity_threshold)[0]
    
    if len(relevant_indices) == 0:
        logger.info(f"Tidak ada dokumen dengan similarity >= {similarity_threshold}")
        return []
    
    # Urutkan berdasarkan similarity (tertinggi ke terendah)
    sorted_indices = relevant_indices[np.argsort(similarities[relevant_indices])[::-1]]
    top_indices = sorted_indices[:top_k_max]
    
    # Buat hasil dengan konteks dan metadata
    results = []
    for idx in top_indices:
        doc_idx = valid_doc_indices[idx]
        similarity_score = float(similarities[idx])
        
        # Ambil konteks dengan dokumen sekitarnya
        start = max(0, doc_idx - 1)
        end = min(len(docs), doc_idx + 2)
        context = ' '.join(docs[start:end]).strip()
        
        results.append({
            "text": context,
            "similarity": similarity_score,
            "doc_index": doc_idx,
            "context_range": f"{start}-{end-1}"
        })
    
    return results

def get_model_params(model_type: str) -> dict:
    """Parameter model yang dioptimalkan"""
    base_params = {
        "vector_size": EMBEDDING_DIM,
        "window": 5,
        "min_count": 1,  # Dikurangi untuk dataset kecil
        "workers": min(4, os.cpu_count() or 1),
        "epochs": 15,  # Ditingkatkan untuk akurasi
        "sg": 1,  # Skip-gram untuk performa lebih baik
        "hs": 0,  # Negative sampling
        "negative": 10
    }
    
    if model_type == "fasttext":
        base_params.update({
            "bucket": 50000,  # Dikurangi untuk efisiensi
            "min_n": 3,
            "max_n": 6
        })
    
    return base_params

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Document Search API",
        "version": "1.0.0",
        "models_cached": len(model_cache)
    }

@app.get("/models/{userId}/{chatbotId}")
async def list_models(userId: str, chatbotId: str):
    """List semua model untuk user dan chatbot tertentu"""
    try:
        user_path = Path("model") / userId / chatbotId
        
        if not user_path.exists():
            return JSONResponse({"models": []})
        
        models = []
        for pdf_dir in user_path.iterdir():
            if pdf_dir.is_dir():
                metadata_file = pdf_dir / "metadata.json"
                if metadata_file.exists():
                    try:
                        with open(metadata_file, "r") as f:
                            metadata = json.load(f)
                        
                        # Check model files
                        word2vec_exists = (pdf_dir / "word2vec.model").exists()
                        fasttext_exists = (pdf_dir / "fasttext.model").exists()
                        
                        models.append({
                            "pdfTitle": pdf_dir.name,
                            "metadata": metadata,
                            "available_models": {
                                "word2vec": word2vec_exists,
                                "fasttext": fasttext_exists
                            }
                        })
                    except Exception as e:
                        logger.warning(f"Error reading metadata for {pdf_dir}: {e}")
        
        return JSONResponse({"models": models})
        
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        raise HTTPException(500, "Error listing models")

@app.post("/train")
async def train_model(
    pdf: UploadFile = File(...),
    userId: str = Form(...),
    chatbotId: str = Form(...),
    modelType: str = Form(..., regex="^(word2vec|fasttext)$"),
    pdfTitle: str = Form(...),
):
    """Training model dengan error handling yang lebih baik"""
    start_time = time.time()
    logger.info(f"Training started - User: {userId}, Chatbot: {chatbotId}, PDF: {pdfTitle}")

    # Validasi input
    if not pdf.filename or not pdf.filename.lower().endswith('.pdf'):
        raise HTTPException(400, "Hanya file PDF yang diterima")
    
    # Validasi parameter
    if not all([userId, chatbotId, pdfTitle]):
        raise HTTPException(400, "Parameter userId, chatbotId, dan pdfTitle harus diisi")

    tmp_path = None
    try:
        # Baca dan validasi file
        content = await pdf.read()
        if not content:
            raise HTTPException(400, "File PDF kosong")
            
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(413, f"Ukuran file melebihi {MAX_FILE_SIZE/1024/1024:.1f}MB")

        # Simpan ke temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(content)
            tmp_path = tmp_file.name

        # Load dan split dokumen
        loader = PyMuPDFLoader(tmp_path)
        raw_docs = loader.load()

        if not raw_docs:
            raise HTTPException(400, "PDF tidak dapat dibaca atau kosong")

        # Split dokumen dengan parameter yang dioptimalkan
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=MAX_CHUNK_SIZE,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", "! ", "? ", " "],
            keep_separator=True
        )
        
        split_docs = text_splitter.split_documents(raw_docs)
        
        # Cleaning dan validasi dokumen
        cleaned_docs = []
        for doc in split_docs:
            try:
                text = doc.page_content.strip()
                text = re.sub(r'\s+', ' ', text)
                
                if len(text) >= MIN_CHUNK_LENGTH:
                    cleaned_docs.append(text)
            except Exception as e:
                logger.warning(f"Error cleaning document: {str(e)}")
                
        if not cleaned_docs:
            raise HTTPException(400, "Tidak ada teks yang valid dalam PDF")

        logger.info(f"Successfully processed {len(cleaned_docs)} document chunks")

        # Preprocessing untuk training
        processed_docs = []
        for doc in cleaned_docs:
            tokens = preprocess_text(doc)
            if len(tokens) >= 3:  # Minimal 3 token per dokumen
                processed_docs.append(tokens)
        
        if not processed_docs:
            raise HTTPException(400, "Gagal melakukan preprocessing - tidak ada token yang valid")

        # Training model
        model_params = get_model_params(modelType)
        
        logger.info(f"Training {modelType} with {len(processed_docs)} documents")
        
        if modelType == 'word2vec':
            model = Word2Vec(processed_docs, **model_params)
        else:
            model = FastText(processed_docs, **model_params)

        # Simpan hasil
        base_path = Path("model") / userId / chatbotId / pdfTitle
        storage_path = Path("storage") / userId / chatbotId / pdfTitle
        
        base_path.mkdir(parents=True, exist_ok=True)
        storage_path.mkdir(parents=True, exist_ok=True)

        # Simpan model
        model_file = base_path / f"{modelType}.model"
        model.save(str(model_file))

        # Hitung statistik
        vocab_size = len(model.wv.key_to_index)
        avg_length = sum(len(d) for d in cleaned_docs) // len(cleaned_docs) if cleaned_docs else 0

        # Metadata
        metadata = {
            "model_type": modelType,
            "created_at": time.time(),
            "pdf_title": pdfTitle,
            "parameters": model_params,
            "statistics": {
                "total_chunks": len(cleaned_docs),
                "processed_chunks": len(processed_docs),
                "avg_chunk_length": avg_length,
                "vocabulary_size": vocab_size,
                "training_time": time.time() - start_time
            }
        }
        
        # Simpan metadata
        with open(base_path / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        # Simpan dokumen
        with open(storage_path / "original_texts.txt", "w", encoding="utf-8") as f:
            for doc in cleaned_docs:
                f.write(doc + '\n---\n')

        # Cache model
        cache_key = get_model_cache_key(userId, chatbotId, pdfTitle, modelType)
        if len(model_cache) < 10:
            model_cache[cache_key] = model

        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")

        return JSONResponse({
            "status": "success",
            "message": "Model berhasil dilatih",
            "data": {
                "model_type": modelType,
                "statistics": metadata["statistics"],
                "model_path": str(base_path)
            }
        }, status_code=201)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Training error: {str(e)}", exc_info=True)
        raise HTTPException(500, f"Error during training: {str(e)}")
    
    finally:
        # Cleanup temporary file
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception as e:
                logger.error(f"Failed to remove temp file: {str(e)}")

@app.post("/query")
async def query_model(
    query: str = Form(...),
    userId: str = Form(...),
    chatbotId: str = Form(...),
    modelType: str = Form(..., regex="^(word2vec|fasttext)$"),
    pdfTitle: str = Form(...),
    topK: int = Form(default=5, ge=1, le=20),
    similarityThreshold: float = Form(default=0.3, ge=0.0, le=1.0),
):
    """Query model dengan validasi dan error handling yang lebih baik"""
    
    if not query.strip():
        raise HTTPException(400, "Query tidak boleh kosong")
    
    try:
        # Konstruksi path
        base_path = Path("model") / userId / chatbotId / pdfTitle
        storage_path = Path("storage") / userId / chatbotId / pdfTitle
        
        if not base_path.exists() or not storage_path.exists():
            raise HTTPException(404, "Model atau dokumen tidak ditemukan")
        
        # Load metadata
        metadata_file = base_path / "metadata.json"
        if not metadata_file.exists():
            raise HTTPException(404, "Metadata model tidak ditemukan")
            
        with open(metadata_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        # Load model
        model_file = base_path / f"{modelType}.model"
        if not model_file.exists():
            raise HTTPException(404, f"Model {modelType} tidak ditemukan")
        
        cache_key = get_model_cache_key(userId, chatbotId, pdfTitle, modelType)
        model = load_model_cached(model_file, modelType, cache_key)

        # Load dokumen
        text_file = storage_path / "original_texts.txt"
        if not text_file.exists():
            raise HTTPException(404, "Dokumen referensi tidak ditemukan")
            
        with open(text_file, "r", encoding="utf-8") as f:
            content = f.read()
            documents = [doc.strip() for doc in content.split('\n---\n') if doc.strip()]

        if not documents:
            raise HTTPException(404, "Tidak ada dokumen valid yang ditemukan")

        # Proses query
        logger.info(f"Processing query: '{query[:50]}...' for {len(documents)} documents")
        
        results = find_context(
            query=query,
            embedding_model=model,
            docs=documents,
            top_k_max=topK,
            similarity_threshold=similarityThreshold
        )

        return JSONResponse({
            "status": "success",
            "query": query,
            "results": results,
            "metadata": {
                "model_type": modelType,
                "documents_count": len(documents),
                "results_count": len(results),
                "threshold_used": similarityThreshold,
                "model_info": metadata.get("statistics", {})
            }
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query processing error: {str(e)}", exc_info=True)
        raise HTTPException(500, f"Error processing query: {str(e)}")

@app.delete("/models/{userId}/{chatbotId}/{pdfTitle}")
async def delete_model(userId: str, chatbotId: str, pdfTitle: str):
    """Hapus model dan dokumen terkait"""
    try:
        base_path = Path("model") / userId / chatbotId / pdfTitle
        storage_path = Path("storage") / userId / chatbotId / pdfTitle
        
        deleted_files = []
        
        # Hapus dari cache
        for model_type in ['word2vec', 'fasttext']:
            cache_key = get_model_cache_key(userId, chatbotId, pdfTitle, model_type)
            if cache_key in model_cache:
                del model_cache[cache_key]
                logger.info(f"Removed from cache: {cache_key}")
        
        # Hapus direktori model
        if base_path.exists():
            import shutil
            shutil.rmtree(base_path)
            deleted_files.append(str(base_path))
        
        # Hapus direktori storage
        if storage_path.exists():
            import shutil
            shutil.rmtree(storage_path)
            deleted_files.append(str(storage_path))
        
        if not deleted_files:
            raise HTTPException(404, "Model tidak ditemukan")
        
        return JSONResponse({
            "status": "success",
            "message": "Model berhasil dihapus",
            "deleted_paths": deleted_files
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting model: {str(e)}")
        raise HTTPException(500, f"Error deleting model: {str(e)}")

@app.post("/train/proposed-model")
async def train_proposed_model(
    pdf: UploadFile = File(...),
    userId: str = Form(...),
    chatbotId: str = Form(...),
    pdfTitle: str = Form(...),
    modelType: str = Form(default="fasttext", regex="^(word2vec|fasttext)$")
):
    """Training model untuk proposed hybrid approach"""
    start_time = time.time()
    logger.info(f"Proposed model training started - User: {userId}, Chatbot: {chatbotId}, PDF: {pdfTitle}")

    # Validasi input (sama seperti endpoint train biasa)
    if not pdf.filename or not pdf.filename.lower().endswith('.pdf'):
        raise HTTPException(400, "Hanya file PDF yang diterima")
    
    if not all([userId, chatbotId, pdfTitle]):
        raise HTTPException(400, "Parameter userId, chatbotId, dan pdfTitle harus diisi")

    tmp_path = None
    try:
        # Baca dan validasi file
        content = await pdf.read()
        if not content:
            raise HTTPException(400, "File PDF kosong")
            
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(413, f"Ukuran file melebihi {MAX_FILE_SIZE/1024/1024:.1f}MB")

        # Simpan ke temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(content)
            tmp_path = tmp_file.name

        # Load dan split dokumen
        loader = PyMuPDFLoader(tmp_path)
        raw_docs = loader.load()

        if not raw_docs:
            raise HTTPException(400, "PDF tidak dapat dibaca atau kosong")

        # Split dokumen dengan parameter yang dioptimalkan untuk hybrid search
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,  # Sedikit lebih besar untuk context yang lebih baik
            chunk_overlap=150,
            separators=["\n\n", "\n", ". ", "! ", "? ", " "],
            keep_separator=True
        )
        
        split_docs = text_splitter.split_documents(raw_docs)
        
        # Cleaning dan validasi dokumen
        cleaned_docs = []
        for doc in split_docs:
            try:
                text = doc.page_content.strip()
                text = re.sub(r'\s+', ' ', text)
                
                if len(text) >= MIN_CHUNK_LENGTH:
                    cleaned_docs.append(text)
            except Exception as e:
                logger.warning(f"Error cleaning document: {str(e)}")
                
        if not cleaned_docs:
            raise HTTPException(400, "Tidak ada teks yang valid dalam PDF")

        logger.info(f"Successfully processed {len(cleaned_docs)} document chunks")

        # Preprocessing untuk training FastText/Word2Vec
        processed_docs = []
        for doc in cleaned_docs:
            tokens = preprocess_text(doc)
            if len(tokens) >= 3:
                processed_docs.append(tokens)
        
        if not processed_docs:
            raise HTTPException(400, "Gagal melakukan preprocessing - tidak ada token yang valid")

        # Training FastText/Word2Vec model
        model_params = get_model_params(modelType)
        
        logger.info(f"Training {modelType} with {len(processed_docs)} documents")
        
        if modelType == 'word2vec':
            semantic_model = Word2Vec(processed_docs, **model_params)
        else:
            semantic_model = FastText(processed_docs, **model_params)

        # Buat BM25 index untuk keyword matching
        logger.info("Creating BM25 index...")
        bm25_cache_key = f"{userId}_{chatbotId}_{pdfTitle}_bm25"
        try:
            bm25_index = create_bm25_index(cleaned_docs, bm25_cache_key)
        except Exception as e:
            logger.error(f"Error creating BM25 index: {str(e)}")
            raise HTTPException(500, "Error creating BM25 index")

        # Simpan hasil
        base_path = Path("model") / userId / chatbotId / pdfTitle
        storage_path = Path("storage") / userId / chatbotId / pdfTitle
        
        base_path.mkdir(parents=True, exist_ok=True)
        storage_path.mkdir(parents=True, exist_ok=True)

        # Simpan semantic model
        semantic_model_file = base_path / f"{modelType}_proposed.model"
        semantic_model.save(str(semantic_model_file))

        # Simpan BM25 index (serialize as pickle)
        import pickle
        bm25_file = base_path / "bm25_index.pkl"
        with open(bm25_file, 'wb') as f:
            pickle.dump(bm25_index, f)

        # Hitung statistik
        vocab_size = len(semantic_model.wv.key_to_index)
        avg_length = sum(len(d) for d in cleaned_docs) // len(cleaned_docs) if cleaned_docs else 0

        # Test sample queries untuk validasi
        sample_queries = [
            "apa yang harus saya lakukan untuk setup tipe room?",
            "bagaimana cara mengatur harga kamar?",
            "jelaskan apa itu advertise pop-up!",
        ]
        
        test_results = {}
        for sample_query in sample_queries:
            try:
                complexity = analyze_query_complexity(sample_query)
                weights = get_weight_strategy(complexity)
                hybrid_results = hybrid_search(
                    sample_query, semantic_model, bm25_index, cleaned_docs[:10], weights, 3
                )
                test_results[sample_query] = {
                    "complexity": complexity.query_type,
                    "weights": weights,
                    "results_count": len(hybrid_results)
                }
            except Exception as e:
                test_results[sample_query] = {"error": str(e)}

        # Enhanced metadata
        metadata = {
            "model_type": f"{modelType}_hybrid",
            "created_at": time.time(),
            "pdf_title": pdfTitle,
            "semantic_parameters": model_params,
            "hybrid_features": {
                "semantic_model": modelType,
                "keyword_matching": "BM25",
                "context_scoring": True,
                "mmr_reranking": True
            },
            "statistics": {
                "total_chunks": len(cleaned_docs),
                "processed_chunks": len(processed_docs),
                "avg_chunk_length": avg_length,
                "vocabulary_size": vocab_size,
                "training_time": time.time() - start_time,
                "bm25_terms": len(bm25_index.doc_freqs) if hasattr(bm25_index, 'doc_freqs') else 0
            },
            "validation": {
                "sample_queries_tested": len(sample_queries),
                "test_results": test_results
            }
        }
        
        # Simpan metadata
        with open(base_path / "metadata_proposed.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        # Simpan dokumen dengan format yang sesuai untuk hybrid search
        with open(storage_path / "documents_proposed.txt", "w", encoding="utf-8") as f:
            for i, doc in enumerate(cleaned_docs):
                f.write(f"DOC_{i}||{doc}\n")

        # Cache models
        semantic_cache_key = f"{userId}_{chatbotId}_{pdfTitle}_{modelType}_proposed"
        if len(model_cache) < 10:
            model_cache[semantic_cache_key] = semantic_model

        training_time = time.time() - start_time
        logger.info(f"Proposed model training completed in {training_time:.2f} seconds")

        return JSONResponse({
            "status": "success",
            "message": "Hybrid model berhasil dilatih",
            "data": {
                "model_type": f"{modelType}_hybrid",
                "features": metadata["hybrid_features"],
                "statistics": metadata["statistics"],
                "validation": metadata["validation"],
                "model_path": str(base_path)
            }
        }, status_code=201)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Proposed model training error: {str(e)}", exc_info=True)
        raise HTTPException(500, f"Error during hybrid model training: {str(e)}")
    
    finally:
        # Cleanup temporary file
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception as e:
                logger.error(f"Failed to remove temp file: {str(e)}")

@app.post("/query/proposed-model")
async def query_proposed_model(
    query: str = Form(...),
    userId: str = Form(...),
    chatbotId: str = Form(...),
    pdfTitle: str = Form(...),
    modelType: str = Form(default="fasttext", regex="^(word2vec|fasttext)$"),
    topK: int = Form(default=5, ge=1, le=20),
    similarityThreshold: float = Form(default=0.2, ge=0.0, le=1.0),
    mmrLambda: float = Form(default=0.7, ge=0.0, le=1.0),
    useGPT: bool = Form(default=False),
    gptModel: str = Form(default="gpt-3.5-turbo"),
    includeRAGAS: bool = Form(default=False),
    promptTemplate: Optional[str] = Form(default=None),
    maxToken: Optional[int] = Form(default=500, ge=100, le=2000),
    temperature: Optional[float] = Form(default=0.7, ge=0.0, le=1.0)
):
    """Query menggunakan proposed hybrid model dengan semua fitur advanced"""
    
    if not query.strip():
        raise HTTPException(400, "Query tidak boleh kosong")
    
    start_time = time.time()
    
    try:
        # Konstruksi path
        base_path = Path("model") / userId / chatbotId / pdfTitle
        storage_path = Path("storage") / userId / chatbotId / pdfTitle
        
        if not base_path.exists() or not storage_path.exists():
            raise HTTPException(404, "Model atau dokumen tidak ditemukan")
        
        # Load metadata
        metadata_file = base_path / "metadata_proposed.json"
        if not metadata_file.exists():
            raise HTTPException(404, "Proposed model metadata tidak ditemukan")
            
        with open(metadata_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        # 1. Query Analysis
        logger.info(f"Analyzing query complexity: '{query[:50]}...'")
        complexity = analyze_query_complexity(query)
        weights = get_weight_strategy(complexity)
        
        # 2. Load semantic model
        semantic_model_file = base_path / f"{modelType}_proposed.model"
        if not semantic_model_file.exists():
            raise HTTPException(404, f"Proposed {modelType} model tidak ditemukan")
        
        semantic_cache_key = f"{userId}_{chatbotId}_{pdfTitle}_{modelType}_proposed"
        semantic_model = load_model_cached(semantic_model_file, modelType, semantic_cache_key)

        # 3. Load BM25 index
        bm25_file = base_path / "bm25_index.pkl"
        if not bm25_file.exists():
            raise HTTPException(404, "BM25 index tidak ditemukan")
        
        import pickle
        bm25_cache_key = f"{userId}_{chatbotId}_{pdfTitle}_bm25"
        
        if bm25_cache_key in bm25_cache:
            bm25_index = bm25_cache[bm25_cache_key]
        else:
            with open(bm25_file, 'rb') as f:
                bm25_index = pickle.load(f)
            if len(bm25_cache) < 10:
                bm25_cache[bm25_cache_key] = bm25_index

        # 4. Load documents
        documents_file = storage_path / "documents_proposed.txt"
        if not documents_file.exists():
            raise HTTPException(404, "Dokumen proposed tidak ditemukan")
            
        documents = []
        with open(documents_file, "r", encoding="utf-8") as f:
            for line in f:
                if '||' in line:
                    _, doc_text = line.strip().split('||', 1)
                    documents.append(doc_text)

        if not documents:
            raise HTTPException(404, "Tidak ada dokumen valid yang ditemukan")

        # 5. Hybrid Search
        logger.info(f"Performing hybrid search with weights: {weights}")
        search_results = hybrid_search(
            query=query,
            fasttext_model=semantic_model,
            bm25_index=bm25_index,
            documents=documents,
            weights=weights,
            top_k=min(topK * 2, 20)  # Ambil lebih banyak untuk MMR
        )

        if not search_results:
            return JSONResponse({
                "status": "success",
                "query": query,
                "complexity_analysis": {
                    "type": complexity.query_type,
                    "score": complexity.complexity_score,
                    "weights_used": weights
                },
                "results": [],
                "message": "Tidak ditemukan dokumen yang relevan"
            })

        # 6. MMR Reranking
        logger.info(f"Performing MMR reranking with lambda={mmrLambda}")
        mmr_results = mmr_reranking(
            results=search_results,
            lambda_param=mmrLambda,
            top_k=topK
        )

        # 7. GPT Generation (optional)
        gpt_response = None
        if useGPT and mmr_results:
            try:
                logger.info("Generating answer with GPT")
                if not promptTemplate:
                    promptTemplate = DEFAULT_SYSTEM_PROMPT

                gpt_response = await generate_answer_with_gpt(
                    query=query,
                    contexts=mmr_results,
                    model=gptModel,
                    system_prompt=promptTemplate,
                    max_tokens=maxToken,
                    temperature=temperature
                )

            except Exception as e:
                logger.error(f"GPT generation failed: {str(e)}")
                gpt_response = {"error": str(e)}

        # 8. RAGAS Evaluation (optional)
        ragas_metrics = None
        if includeRAGAS and gpt_response and "answer" in gpt_response:
            try:
                logger.info("Calculating RAGAS metrics")
                contexts_for_ragas = [result.text for result in mmr_results]
                ragas_metrics = calculate_ragas_metrics(
                    query=query,
                    contexts=contexts_for_ragas,
                    answer=gpt_response["answer"]
                )
            except Exception as e:
                logger.error(f"RAGAS calculation failed: {str(e)}")
                ragas_metrics = {"error": str(e)}

        # 9. Prepare response
        processing_time = time.time() - start_time

        # Format results for response
        formatted_results = []
        for i, result in enumerate(mmr_results):
            # Find original search result for detailed scores
            original_result = None
            for sr in search_results:
                if sr.doc_index == result.doc_index:
                    original_result = sr
                    break
            
            formatted_result = {
                "rank": i + 1,
                "text": result.text,
                "doc_index": result.doc_index,
                "final_score": result.final_score,
                "diversity_penalty": result.diversity_penalty,
                "original_rank": result.original_rank + 1
            }
            
            if original_result:
                formatted_result.update({
                    "detailed_scores": {
                        "fasttext_similarity": original_result.fasttext_similarity,
                        "bm25_score": original_result.bm25_score,
                        "context_score": original_result.context_score,
                        "weighted_score": original_result.weighted_score
                    },
                    "context_range": original_result.context_range
                })
            
            formatted_results.append(formatted_result)

        response_data = {
            "status": "success",
            "query": query,
            "processing_time": processing_time,
            "complexity_analysis": {
                "type": complexity.query_type,
                "score": complexity.complexity_score,
                "word_count": complexity.word_count,
                "unique_words": complexity.unique_words,
                "question_words": complexity.question_words,
                "weights_used": weights
            },
            "search_pipeline": {
                "hybrid_search_results": len(search_results),
                "mmr_reranked_results": len(mmr_results),
                "mmr_lambda": mmrLambda,
                "similarity_threshold": similarityThreshold
            },
            "results": formatted_results,
            "metadata": {
                "model_type": f"{modelType}_hybrid",
                "documents_count": len(documents),
                "features_used": {
                    "semantic_search": True,
                    "keyword_search": True,
                    "context_scoring": True,
                    "mmr_reranking": True,
                    "gpt_generation": useGPT,
                    "ragas_evaluation": includeRAGAS
                }
            }
        }

        # Add GPT response if available
        if gpt_response:
            response_data["gpt_generation"] = gpt_response

        # Add RAGAS metrics if available
        if ragas_metrics is not None:
            if isinstance(ragas_metrics, RAGASMetrics):
                response_data["ragas_evaluation"] = {
                    "context_relevance": ragas_metrics.context_relevance,
                    "faithfulness": ragas_metrics.faithfulness,
                    "answer_relevance": ragas_metrics.answer_relevance,
                    "overall_score": ragas_metrics.overall_score
                }
            elif isinstance(ragas_metrics, dict):
                response_data["ragas_evaluation"] = ragas_metrics
            else:
                logger.warning(f"Unexpected type for ragas_metrics: {type(ragas_metrics)}")

        return JSONResponse(response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Proposed query processing error: {str(e)}", exc_info=True)
        raise HTTPException(500, f"Error processing proposed query: {str(e)}")

@app.get("/models/{userId}/{chatbotId}/proposed")
async def list_proposed_models(userId: str, chatbotId: str):
    """List semua proposed models untuk user dan chatbot tertentu"""
    try:
        user_path = Path("model") / userId / chatbotId
        
        if not user_path.exists():
            return JSONResponse({"proposed_models": []})
        
        models = []
        for pdf_dir in user_path.iterdir():
            if pdf_dir.is_dir():
                metadata_file = pdf_dir / "metadata_proposed.json"
                if metadata_file.exists():
                    try:
                        with open(metadata_file, "r") as f:
                            metadata = json.load(f)
                        
                        # Check proposed model files
                        word2vec_exists = (pdf_dir / "word2vec_proposed.model").exists()
                        fasttext_exists = (pdf_dir / "fasttext_proposed.model").exists()
                        bm25_exists = (pdf_dir / "bm25_index.pkl").exists()
                        
                        models.append({
                            "pdfTitle": pdf_dir.name,
                            "metadata": metadata,
                            "available_components": {
                                "word2vec_semantic": word2vec_exists,
                                "fasttext_semantic": fasttext_exists,
                                "bm25_keyword": bm25_exists
                            },
                            "hybrid_features": metadata.get("hybrid_features", {}),
                            "validation_results": metadata.get("validation", {})
                        })
                    except Exception as e:
                        logger.warning(f"Error reading proposed metadata for {pdf_dir}: {e}")
        
        return JSONResponse({"proposed_models": models})
        
    except Exception as e:
        logger.error(f"Error listing proposed models: {str(e)}")
        raise HTTPException(500, "Error listing proposed models")

@app.delete("/models/{userId}/{chatbotId}/{pdfTitle}/proposed")
async def delete_proposed_model(userId: str, chatbotId: str, pdfTitle: str):
    """Hapus proposed model dan komponen terkait"""
    try:
        base_path = Path("model") / userId / chatbotId / pdfTitle
        storage_path = Path("storage") / userId / chatbotId / pdfTitle
        
        deleted_files = []
        
        # Hapus dari cache
        for model_type in ['word2vec', 'fasttext']:
            cache_key = f"{userId}_{chatbotId}_{pdfTitle}_{model_type}_proposed"
            if cache_key in model_cache:
                del model_cache[cache_key]
                logger.info(f"Removed from model cache: {cache_key}")
        
        # Hapus BM25 dari cache
        bm25_cache_key = f"{userId}_{chatbotId}_{pdfTitle}_bm25"
        if bm25_cache_key in bm25_cache:
            del bm25_cache[bm25_cache_key]
            logger.info(f"Removed from BM25 cache: {bm25_cache_key}")
        
        # Hapus file proposed models
        proposed_files = [
            "word2vec_proposed.model",
            "fasttext_proposed.model", 
            "bm25_index.pkl",
            "metadata_proposed.json"
        ]
        
        for filename in proposed_files:
            file_path = base_path / filename
            if file_path.exists():
                file_path.unlink()
                deleted_files.append(str(file_path))
        
        # Hapus dokumen proposed
        proposed_docs = storage_path / "documents_proposed.txt"
        if proposed_docs.exists():
            proposed_docs.unlink()
            deleted_files.append(str(proposed_docs))
        
        if not deleted_files:
            raise HTTPException(404, "Proposed model tidak ditemukan")
        
        return JSONResponse({
            "status": "success",
            "message": "Proposed model berhasil dihapus",
            "deleted_files": deleted_files
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting proposed model: {str(e)}")
        raise HTTPException(500, f"Error deleting proposed model: {str(e)}")

@app.post("/train/baseline-model")
async def train_baseline_model(
    pdf: UploadFile = File(...),
    userId: str = Form(...),
    chatbotId: str = Form(...),
    pdfTitle: str = Form(...),
    # Hyperparameter tuning parameters sesuai diagram
    vectorSize: int = Form(default=300),  # Dimensi 300 sesuai diagram
    learningRate: float = Form(default=0.05, ge=0.001, le=1.0),
    epochs: int = Form(default=20, ge=5, le=100),
    windowSize: int = Form(default=5, ge=1, le=10),
    minCount: int = Form(default=1, ge=1, le=10)
):
    """Training baseline model sesuai diagram - FastText dengan hyperparameter tuning"""
    start_time = time.time()
    logger.info(f"Baseline model training started - User: {userId}, Chatbot: {chatbotId}, PDF: {pdfTitle}")

    # Validasi input
    if not pdf.filename or not pdf.filename.lower().endswith('.pdf'):
        raise HTTPException(400, "Hanya file PDF yang diterima")
    
    if not all([userId, chatbotId, pdfTitle]):
        raise HTTPException(400, "Parameter userId, chatbotId, dan pdfTitle harus diisi")

    tmp_path = None
    try:
        # Baca dan validasi file
        content = await pdf.read()
        if not content:
            raise HTTPException(400, "File PDF kosong")
            
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(413, f"Ukuran file melebihi {MAX_FILE_SIZE/1024/1024:.1f}MB")

        # Simpan ke temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(content)
            tmp_path = tmp_file.name

        # Load dan split dokumen - optimized untuk baseline model
        loader = PyMuPDFLoader(tmp_path)
        raw_docs = loader.load()

        if not raw_docs:
            raise HTTPException(400, "PDF tidak dapat dibaca atau kosong")

        # Split dokumen dengan parameter baseline (sesuai diagram: Document Chunks)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,  # Ukuran chunk standar untuk baseline
            chunk_overlap=150,  # Minimal overlap untuk baseline
            separators=["\n\n", "\n", ". ", "! ", "? ", " "],
            keep_separator=False  # Baseline tidak perlu separator
        )
        
        split_docs = text_splitter.split_documents(raw_docs)
        
        # Cleaning dokumen dengan preprocessing sesuai diagram
        cleaned_docs = []
        for doc in split_docs:
            try:
                text = doc.page_content.strip()
                # Case folding (sesuai diagram preprocessing)
                text = text.lower()
                # Normalize whitespace
                text = re.sub(r'\s+', ' ', text)
                
                if len(text) >= 30:  # Minimum length untuk baseline
                    cleaned_docs.append(text)
            except Exception as e:
                logger.warning(f"Error cleaning document: {str(e)}")
                
        if not cleaned_docs:
            raise HTTPException(400, "Tidak ada teks yang valid dalam PDF")

        logger.info(f"Successfully processed {len(cleaned_docs)} document chunks")

        # Preprocessing untuk training (sesuai diagram: Case Folding + Stopword Removal)
        processed_docs = []
        for doc in cleaned_docs:
            tokens = preprocess_text(doc)  # Sudah include case folding & stopword removal
            if len(tokens) >= 3:
                processed_docs.append(tokens)
        
        if not processed_docs:
            raise HTTPException(400, "Gagal melakukan preprocessing - tidak ada token yang valid")

        # Hyperparameter tuning untuk baseline model (sesuai diagram)
        baseline_params = {
            "vector_size": vectorSize,  # Dimensi 300 sesuai diagram
            "window": windowSize,
            "min_count": minCount,
            "workers": min(4, os.cpu_count() or 1),
            "epochs": epochs,
            "alpha": learningRate,  # Learning rate
            "min_alpha": learningRate * 0.0001,  # Final learning rate
            "sg": 1,  # Skip-gram untuk FastText baseline
            "hs": 0,  # Negative sampling
            "negative": 10,
            # FastText specific parameters untuk baseline
            "bucket": 100000,
            "min_n": 3,
            "max_n": 6
        }
        
        logger.info(f"Training FastText baseline with hyperparameters: {baseline_params}")
        
        # Training FastText model (sesuai diagram: FastText Embedding)
        baseline_model = FastText(processed_docs, **baseline_params)

        # Simpan hasil
        base_path = Path("model") / userId / chatbotId / pdfTitle
        storage_path = Path("storage") / userId / chatbotId / pdfTitle
        
        base_path.mkdir(parents=True, exist_ok=True)
        storage_path.mkdir(parents=True, exist_ok=True)

        # Simpan baseline model
        baseline_model_file = base_path / "fasttext_baseline.model"
        baseline_model.save(str(baseline_model_file))

        # Hitung statistik untuk evaluasi
        vocab_size = len(baseline_model.wv.key_to_index)
        avg_length = sum(len(d) for d in cleaned_docs) // len(cleaned_docs) if cleaned_docs else 0

        # Pre-compute embeddings untuk semua dokumen (sesuai diagram: Embedding Document Chunks)
        logger.info("Pre-computing document embeddings for baseline model...")
        doc_embeddings = []
        valid_doc_indices = []
        
        for i, doc in enumerate(cleaned_docs):
            processed_doc = preprocess_text(doc)
            doc_embedding = calculate_embedding(processed_doc, baseline_model)
            
            if doc_embedding is not None:
                doc_embeddings.append(doc_embedding.tolist())  # Convert to list for JSON
                valid_doc_indices.append(i)

        # Test dengan sample queries untuk validasi baseline
        sample_queries = [
            "apa yang harus saya lakukan untuk setup tipe room?",
            "bagaimana cara mengatur harga kamar?",
            "jelaskan apa itu advertise pop-up!",
        ]
        
        baseline_test_results = {}
        for sample_query in sample_queries:
            try:
                # Test cosine similarity calculation
                processed_query = preprocess_text(sample_query)
                query_embedding = calculate_embedding(processed_query, baseline_model)
                
                if query_embedding is not None and doc_embeddings:
                    # Convert back to numpy for similarity calculation
                    doc_emb_array = np.array(doc_embeddings)
                    similarities = cosine_similarity([query_embedding], doc_emb_array)[0]
                    
                    # Top-k retrieval test
                    top_indices = np.argsort(similarities)[::-1][:3]
                    top_scores = similarities[top_indices]
                    
                    baseline_test_results[sample_query] = {
                        "top_similarities": top_scores.tolist(),
                        "avg_similarity": float(np.mean(similarities)),
                        "max_similarity": float(np.max(similarities))
                    }
                else:
                    baseline_test_results[sample_query] = {"error": "No valid embedding"}
                    
            except Exception as e:
                baseline_test_results[sample_query] = {"error": str(e)}

        # Baseline model metadata (sesuai diagram dengan hyperparameter info)
        metadata = {
            "model_type": "fasttext_baseline",
            "created_at": time.time(),
            "pdf_title": pdfTitle,
            "baseline_approach": "FastText + Cosine Similarity + Top-k Retrieval",
            "hyperparameters": baseline_params,
            "preprocessing_steps": [
                "Case Folding",
                "Stopword Removal", 
                "Stemming",
                "Tokenization"
            ],
            "statistics": {
                "total_chunks": len(cleaned_docs),
                "processed_chunks": len(processed_docs),
                "valid_embeddings": len(doc_embeddings),
                "avg_chunk_length": avg_length,
                "vocabulary_size": vocab_size,
                "embedding_dimension": vectorSize,
                "training_time": time.time() - start_time
            },
            "baseline_validation": {
                "sample_queries_tested": len(sample_queries),
                "test_results": baseline_test_results,
                "embedding_coverage": len(doc_embeddings) / len(cleaned_docs) if cleaned_docs else 0
            }
        }
        
        # Simpan metadata baseline
        with open(base_path / "metadata_baseline.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        # Simpan pre-computed embeddings untuk baseline
        embeddings_data = {
            "embeddings": doc_embeddings,
            "valid_indices": valid_doc_indices,
            "dimension": vectorSize
        }
        
        with open(base_path / "doc_embeddings_baseline.json", "w", encoding="utf-8") as f:
            json.dump(embeddings_data, f, indent=2)

        # Simpan dokumen untuk baseline
        with open(storage_path / "documents_baseline.txt", "w", encoding="utf-8") as f:
            for i, doc in enumerate(cleaned_docs):
                f.write(f"DOC_{i}||{doc}\n")

        # Cache baseline model
        baseline_cache_key = f"{userId}_{chatbotId}_{pdfTitle}_fasttext_baseline"
        if len(model_cache) < 10:
            model_cache[baseline_cache_key] = baseline_model

        training_time = time.time() - start_time
        logger.info(f"Baseline model training completed in {training_time:.2f} seconds")

        return JSONResponse({
            "status": "success",
            "message": "Baseline model berhasil dilatih",
            "data": {
                "model_type": "fasttext_baseline",
                "approach": "FastText + Cosine Similarity",
                "hyperparameters": baseline_params,
                "statistics": metadata["statistics"],
                "validation": metadata["baseline_validation"],
                "model_path": str(base_path)
            }
        }, status_code=201)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Baseline model training error: {str(e)}", exc_info=True)
        raise HTTPException(500, f"Error during baseline model training: {str(e)}")
    
    finally:
        # Cleanup temporary file
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception as e:
                logger.error(f"Failed to remove temp file: {str(e)}")

@app.post("/query/baseline-model")
async def query_baseline_model(
    query: str = Form(...),
    userId: str = Form(...),
    chatbotId: str = Form(...),
    pdfTitle: str = Form(...),
    topK: int = Form(default=5, ge=1, le=20),
    useGPT: bool = Form(default=False),
    gptModel: str = Form(default="gpt-3.5-turbo"),
    includeRAGAS: bool = Form(default=False),
    promptTemplate: Optional[str] = Form(default=None),
    maxToken: Optional[int] = Form(default=500, ge=100, le=2000),
    temperature: Optional[float] = Form(default=0.7, ge=0.0, le=1.0)
):
    """Query baseline model sesuai diagram - Simple FastText + Cosine Similarity + Top-k"""
    
    if not query.strip():
        raise HTTPException(400, "Query tidak boleh kosong")
    
    start_time = time.time()
    
    try:
        # Konstruksi path
        base_path = Path("model") / userId / chatbotId / pdfTitle
        storage_path = Path("storage") / userId / chatbotId / pdfTitle
        
        if not base_path.exists() or not storage_path.exists():
            raise HTTPException(404, "Baseline model atau dokumen tidak ditemukan")
        
        # Load baseline metadata
        metadata_file = base_path / "metadata_baseline.json"
        if not metadata_file.exists():
            raise HTTPException(404, "Baseline model metadata tidak ditemukan")
            
        with open(metadata_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        # Load baseline FastText model
        baseline_model_file = base_path / "fasttext_baseline.model"
        if not baseline_model_file.exists():
            raise HTTPException(404, "Baseline FastText model tidak ditemukan")
        
        baseline_cache_key = f"{userId}_{chatbotId}_{pdfTitle}_fasttext_baseline"
        baseline_model = load_model_cached(baseline_model_file, "fasttext", baseline_cache_key)

        # Load pre-computed embeddings
        embeddings_file = base_path / "doc_embeddings_baseline.json"
        if not embeddings_file.exists():
            raise HTTPException(404, "Pre-computed embeddings tidak ditemukan")
            
        with open(embeddings_file, "r") as f:
            embeddings_data = json.load(f)
        
        doc_embeddings = np.array(embeddings_data["embeddings"])
        valid_doc_indices = embeddings_data["valid_indices"]

        # Load documents
        documents_file = storage_path / "documents_baseline.txt"
        if not documents_file.exists():
            raise HTTPException(404, "Baseline documents tidak ditemukan")
            
        documents = []
        with open(documents_file, "r", encoding="utf-8") as f:
            for line in f:
                if '||' in line:
                    _, doc_text = line.strip().split('||', 1)
                    documents.append(doc_text)

        if not documents:
            raise HTTPException(404, "Tidak ada dokumen valid yang ditemukan")

        # BASELINE PIPELINE sesuai diagram:
        
        # 1. Preprocessing (Case Folding + Stopword Removal)
        logger.info(f"Preprocessing query (baseline): '{query[:50]}...'")
        processed_query = preprocess_text(query)
        
        if not processed_query:
            raise HTTPException(400, "Query preprocessing menghasilkan token kosong")

        # 2. FastText Embedding untuk query
        logger.info("Calculating query embedding (baseline)")
        query_embedding = calculate_embedding(processed_query, baseline_model)
        
        if query_embedding is None:
            raise HTTPException(400, "Tidak dapat menghitung embedding untuk query")

        # 3. Cosine Similarity Calculation (sesuai diagram)
        logger.info(f"Calculating cosine similarity with {len(doc_embeddings)} documents")
        similarities = cosine_similarity([query_embedding], doc_embeddings)[0]

        # 4. Top-k Retrieval berdasarkan skor similaritas (sesuai diagram)
        logger.info(f"Performing top-{topK} retrieval")
        top_indices = np.argsort(similarities)[::-1][:topK]
        
        baseline_results = []
        for i, idx in enumerate(top_indices):
            original_doc_idx = valid_doc_indices[idx]
            similarity_score = float(similarities[idx])
            
            # Ambil dokumen dengan context minimal (baseline approach)
            doc_text = documents[original_doc_idx]
            
            baseline_results.append({
                "rank": i + 1,
                "text": doc_text,
                "similarity_score": similarity_score,
                "doc_index": original_doc_idx,
                "method": "fasttext_cosine_similarity"
            })

        # 5. ChatGPT Generator dengan konteks ter-retrieve (opsional, sesuai diagram)
        gpt_response = None
        if useGPT and baseline_results:
            try:
                logger.info("Generating answer with ChatGPT (baseline)")
                # Convert baseline results to MMRResult format for compatibility
                mmr_results = [
                    MMRResult(
                        text=result["text"],
                        final_score=result["similarity_score"],
                        diversity_penalty=0.0,
                        original_rank=result["rank"]-1,
                        doc_index=result["doc_index"]
                    ) for result in baseline_results
                ]

                # Gunakan prompt template dari DEFAULT_SYSTEM_PROMPT jika tidak disediakan
                if not promptTemplate:
                    promptTemplate = DEFAULT_SYSTEM_PROMPT

                gpt_response = await generate_answer_with_gpt(
                    query=query,
                    contexts=mmr_results,
                    model=gptModel,
                    system_prompt=promptTemplate,
                    maxToken=maxToken,
                    temperature=temperature
                )

            except Exception as e:
                logger.error(f"ChatGPT generation failed (baseline): {str(e)}")
                gpt_response = {"error": str(e)}

        # 6. RAGAS Evaluation (sesuai diagram)
        ragas_metrics = None
        if includeRAGAS and gpt_response and "answer" in gpt_response:
            try:
                logger.info("Calculating RAGAS metrics (baseline)")
                contexts_for_ragas = [result["text"] for result in baseline_results]
                ragas_metrics = calculate_ragas_metrics(
                    query=query,
                    contexts=contexts_for_ragas,
                    answer=gpt_response["answer"]
                )
            except Exception as e:
                logger.error(f"RAGAS calculation failed (baseline): {str(e)}")
                ragas_metrics = {"error": str(e)}

        # Prepare response
        processing_time = time.time() - start_time

        response_data = {
            "status": "success",
            "query": query,
            "processing_time": processing_time,
            "model_approach": "baseline",
            "pipeline_steps": [
                "Preprocessing (Case Folding + Stopword Removal)",
                "FastText Embedding", 
                "Cosine Similarity Calculation",
                "Top-k Retrieval",
                "ChatGPT Generation (optional)",
                "RAGAS Evaluation (optional)"
            ],
            "results": baseline_results,
            "metadata": {
                "model_type": "fasttext_baseline",
                "documents_count": len(documents),
                "embedding_dimension": metadata["statistics"]["embedding_dimension"],
                "hyperparameters": metadata["hyperparameters"],
                "features_used": {
                    "semantic_search": True,
                    "keyword_search": False,
                    "context_scoring": False,
                    "mmr_reranking": False,
                    "gpt_generation": useGPT,
                    "ragas_evaluation": includeRAGAS
                }
            }
        }

        # Add GPT response if available
        if gpt_response:
            response_data["gpt_generation"] = gpt_response

        # Add RAGAS metrics if available
        if ragas_metrics and not isinstance(ragas_metrics, dict) or "error" not in ragas_metrics:
            response_data["ragas_evaluation"] = {
                "context_relevance": ragas_metrics.context_relevance,
                "faithfulness": ragas_metrics.faithfulness,
                "answer_relevance": ragas_metrics.answer_relevance,
                "overall_score": ragas_metrics.overall_score
            }
        elif ragas_metrics:
            response_data["ragas_evaluation"] = ragas_metrics

        return JSONResponse(response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Baseline query processing error: {str(e)}", exc_info=True)
        raise HTTPException(500, f"Error processing baseline query: {str(e)}")

@app.get("/models/{userId}/{chatbotId}/baseline")
async def list_baseline_models(userId: str, chatbotId: str):
    """List semua baseline models untuk user dan chatbot tertentu"""
    try:
        user_path = Path("model") / userId / chatbotId
        
        if not user_path.exists():
            return JSONResponse({"baseline_models": []})
        
        models = []
        for pdf_dir in user_path.iterdir():
            if pdf_dir.is_dir():
                metadata_file = pdf_dir / "metadata_baseline.json"
                if metadata_file.exists():
                    try:
                        with open(metadata_file, "r") as f:
                            metadata = json.load(f)
                        
                        # Check baseline model files
                        fasttext_exists = (pdf_dir / "fasttext_baseline.model").exists()
                        embeddings_exists = (pdf_dir / "doc_embeddings_baseline.json").exists()
                        
                        models.append({
                            "pdfTitle": pdf_dir.name,
                            "metadata": metadata,
                            "available_components": {
                                "fasttext_model": fasttext_exists,
                                "precomputed_embeddings": embeddings_exists
                            },
                            "baseline_approach": metadata.get("baseline_approach", "FastText + Cosine Similarity"),
                            "hyperparameters": metadata.get("hyperparameters", {}),
                            "validation_results": metadata.get("baseline_validation", {})
                        })
                    except Exception as e:
                        logger.warning(f"Error reading baseline metadata for {pdf_dir}: {e}")
        
        return JSONResponse({"baseline_models": models})
        
    except Exception as e:
        logger.error(f"Error listing baseline models: {str(e)}")
        raise HTTPException(500, "Error listing baseline models")

@app.delete("/models/{userId}/{chatbotId}/{pdfTitle}/baseline")
async def delete_baseline_model(userId: str, chatbotId: str, pdfTitle: str):
    """Hapus baseline model dan komponen terkait"""
    try:
        base_path = Path("model") / userId / chatbotId / pdfTitle
        storage_path = Path("storage") / userId / chatbotId / pdfTitle
        
        deleted_files = []
        
        # Hapus dari cache
        baseline_cache_key = f"{userId}_{chatbotId}_{pdfTitle}_fasttext_baseline"
        if baseline_cache_key in model_cache:
            del model_cache[baseline_cache_key]
            logger.info(f"Removed from cache: {baseline_cache_key}")
        
        # Hapus file baseline models
        baseline_files = [
            "fasttext_baseline.model",
            "metadata_baseline.json",
            "doc_embeddings_baseline.json"
        ]
        
        for filename in baseline_files:
            file_path = base_path / filename
            if file_path.exists():
                file_path.unlink()
                deleted_files.append(str(file_path))
        
        # Hapus dokumen baseline
        baseline_docs = storage_path / "documents_baseline.txt"
        if baseline_docs.exists():
            baseline_docs.unlink()
            deleted_files.append(str(baseline_docs))
        
        if not deleted_files:
            raise HTTPException(404, "Baseline model tidak ditemukan")
        
        return JSONResponse({
            "status": "success",
            "message": "Baseline model berhasil dihapus",
            "deleted_files": deleted_files
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting baseline model: {str(e)}")
        raise HTTPException(500, f"Error deleting baseline model: {str(e)}")

async def compare_models(
    query: str = Form(...),
    userId: str = Form(...),
    chatbotId: str = Form(...),
    pdfTitle: str = Form(...),
    modelType: str = Form(default="fasttext", regex="^(word2vec|fasttext)$"),
    topK: int = Form(default=5, ge=1, le=10)
):
    """Compare hasil antara model biasa dan proposed model"""
    
    if not query.strip():
        raise HTTPException(400, "Query tidak boleh kosong")
            
    try:
        # Test all three models
        results = {}
        
        # 1. Test original model
        try:
            original_response = await query_model(
                query=query,
                userId=userId,
                chatbotId=chatbotId,
                modelType=modelType,
                pdfTitle=pdfTitle,
                topK=topK,
                similarityThreshold=0.4
            )
            results["original_model"] = {
                "status": "success",
                "approach": "Simple FastText/Word2Vec + Cosine Similarity",
                "response_available": True
            }
        except Exception as e:
            results["original_model"] = {"status": "error", "error": str(e)}
        
        # 2. Test baseline model  
        try:
            baseline_response = await query_baseline_model(
                query=query,
                userId=userId,
                chatbotId=chatbotId,
                pdfTitle=pdfTitle,
                topK=topK,
                useGPT=False,
                includeRAGAS=False
            )
            results["baseline_model"] = {
                "status": "success", 
                "approach": "FastText (300D) + Cosine Similarity + Hyperparameter Tuning",
                "response_available": True
            }
        except Exception as e:
            results["baseline_model"] = {"status": "error", "error": str(e)}
        
        # 3. Test proposed model
        try:
            proposed_response = await query_proposed_model(
                query=query,
                userId=userId,
                chatbotId=chatbotId,
                pdfTitle=pdfTitle,
                modelType=modelType,
                topK=topK,
                similarityThreshold=0.2,
                mmrLambda=0.7,
                useGPT=False,
                includeRAGAS=False
            )
            results["proposed_model"] = {
                "status": "success",
                "approach": "Hybrid (FastText + BM25 + Context) + MMR Reranking",
                "response_available": True
            }
        except Exception as e:
            results["proposed_model"] = {"status": "error", "error": str(e)}
        
        # 4. Generate comparison summary
        successful_models = [model for model, result in results.items() if result.get("status") == "success"]
        
        comparison_summary = {
            "query": query,
            "models_compared": ["original", "baseline", "proposed"],
            "successful_models": successful_models,
            "test_parameters": {
                "topK": topK,
                "modelType": modelType
            },
            "model_approaches": {
                "original": "Basic semantic search with minimal preprocessing",
                "baseline": "Optimized FastText with hyperparameter tuning and pre-computed embeddings",
                "proposed": "Advanced hybrid approach with multiple scoring methods and MMR reranking"
            },
            "availability_summary": {
                "original_model": results["original_model"]["status"] == "success",
                "baseline_model": results["baseline_model"]["status"] == "success", 
                "proposed_model": results["proposed_model"]["status"] == "success"
            }
        }
        
        return JSONResponse({
            "status": "success",
            "comparison_results": results,
            "summary": comparison_summary
        })
        
    except Exception as e:
        logger.error(f"Error in model comparison: {str(e)}")
        raise HTTPException(500, f"Error comparing models: {str(e)}")
        
    except Exception as e:
        logger.error(f"Error in model comparison: {str(e)}")
        raise HTTPException(500, f"Error comparing models: {str(e)}")

@app.get("/health/proposed")
async def health_check_proposed():
    """Health check khusus untuk proposed model features"""
    try:
        # Check dependencies
        dependencies_status = {
            "rank_bm25": True,
            "openai": bool(os.getenv("OPENAI_API_KEY")),
            "sklearn": True,
            "numpy": True
        }
        
        # Check caches
        cache_status = {
            "model_cache_size": len(model_cache),
            "bm25_cache_size": len(bm25_cache),
            "cache_limit": 10
        }
        
        # Check available proposed models
        model_count = 0
        try:
            base_model_path = Path("model")
            if base_model_path.exists():
                for user_dir in base_model_path.iterdir():
                    if user_dir.is_dir():
                        for chatbot_dir in user_dir.iterdir():
                            if chatbot_dir.is_dir():
                                for pdf_dir in chatbot_dir.iterdir():
                                    if pdf_dir.is_dir() and (pdf_dir / "metadata_proposed.json").exists():
                                        model_count += 1
        except Exception as e:
            logger.error(f"Error counting proposed models: {e}")
        
        return JSONResponse({
            "status": "healthy",
            "service": "Proposed Model API",
            "version": "1.0.0",
            "features": {
                "hybrid_search": True,
                "mmr_reranking": True,
                "gpt_integration": dependencies_status["openai"],
                "ragas_evaluation": True,
                "query_complexity_analysis": True
            },
            "dependencies": dependencies_status,
            "cache_status": cache_status,
            "proposed_models_count": model_count
        })
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse({
            "status": "error",
            "error": str(e)
        }, status_code=500)

if __name__ == '__main__':
    # Check required dependencies
    try:
        import rank_bm25
        logger.info("BM25 dependency available")
    except ImportError:
        logger.error("rank_bm25 not installed. Install with: pip install rank-bm25")
    
    # Check OpenAI key for GPT features
    if not os.getenv("OPENAI_API_KEY"):
        logger.warning("OPENAI_API_KEY not set. GPT features will be disabled.")
    
    uvicorn.run(
        "api:app",  # Sesuaikan dengan nama file
        host='0.0.0.0',
        port=int(os.getenv('PORT_PY', 8888)),
        reload=os.getenv('ENVIRONMENT', 'production') == 'development',
        log_level="info"
    )