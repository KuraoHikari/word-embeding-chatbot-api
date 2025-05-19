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
from langchain_community.document_loaders import PyMuPDFLoader  # Ganti ke PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
import uvicorn
from pathlib import Path  # Tambahkan ini
import json  # Tambahkan ini di bagian import
import time

# Load environment variables
load_dotenv()

app = FastAPI()

# NLTK Setup
nltk.download('punkt')
nltk.download('stopwords')

# Konfigurasi logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Setup CORS
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


# Konfigurasi
EMBEDDING_DIM = 300
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# Inisialisasi stemmer dan tokenizer
factory = StemmerFactory()
stemmer = factory.create_stemmer()
tokenizer = RegexpTokenizer(r'\w+')

# Preprocessing teks untuk bahasa Indonesia
def preprocess_text(text: str) -> list[str]:
    try:
        # Case folding
        text = text.lower()
        
        # Remove special characters and numbers
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Stemming
        text = stemmer.stem(text)
        
        # Tokenization
        tokens = tokenizer.tokenize(text)
        
        # Stopword removal
        stop_words = stopwords.words('indonesian')
        custom_stopwords = {
            'yg', 'dg', 'dgn', 'ny', 'sih', 'nya', 'kalo', 'deh', 'mah',
            'lah', 'dll', 'tsb', 'dr', 'pd', 'utk', 'sd', 'dpt', 'dlm',
            'thn', 'tgl', 'jd', 'tkr', 'org', 'sbg', 'bs', 'tsb', 'kpd'
        }
        stop_words = set(stop_words).union(custom_stopwords)
        
        # Filtering
        filtered_tokens = [
            word for word in tokens 
            if (word not in stop_words and 
                len(word) > 2 and 
                not any(char.isdigit() for char in word))
        ]
        
        return filtered_tokens
    
    except Exception as e:
        logger.error(f"Preprocessing error: {str(e)}")
        return []

# Fungsi pencarian konteks
def find_context(query, embedding_model, docs, top_k_max=5, similarity_threshold=0.4):
    processed_query = preprocess_text(query)
    query_embedding = np.mean([embedding_model.wv[word] 
                             for word in processed_query if word in embedding_model.wv], axis=0)

    doc_embeddings = []
    for doc in docs:
        processed_doc = preprocess_text(doc)
        doc_embedding = np.mean([embedding_model.wv[word] 
                               for word in processed_doc if word in embedding_model.wv], axis=0)
        doc_embeddings.append(doc_embedding)

    similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
    relevant_indices = np.where(similarities >= similarity_threshold)[0]
    
    if len(relevant_indices) == 0:
        return []
    
    sorted_indices = relevant_indices[np.argsort(similarities[relevant_indices])[::-1]]
    top_indices = sorted_indices[:top_k_max]
    
     # Fungsi untuk mendapatkan konteks dengan dokumen sekitarnya
    def get_context_chunk(index: int) -> str:
        start = max(0, index - 1)
        end = min(len(docs), index + 2)  # +2 karena slicing exclusive
        return ' '.join(docs[start:end])
    
    # Ambil konteks untuk setiap indeks teratas
    return [get_context_chunk(i) for i in top_indices]

def get_model_params(model_type: str):
    """Return parameters spesifik untuk tiap model"""
    base_params = {
        "vector_size": 100,
        "window": 5,
        "min_count": 2,
        "workers": 4,
        "epochs": 10
    }
    
    if model_type == "fasttext":
        base_params["bucket"] = 100000  # Parameter khusus FastText
        base_params["min_n"] = 3
        base_params["max_n"] = 6
    
    return base_params

@app.post("/train")
async def train_model(
    pdf: UploadFile = File(...),
    userId: str = Form(...),
    chatbotId: str = Form(...),
    modelType: str = Form(..., regex="^(word2vec|fasttext)$"),
    pdfTitle: str = Form(...),
):
    start_time = time.time()
    logger.info("Training started")

    try:
        # Validasi awal
        if not pdf.filename.lower().endswith('.pdf'):
            raise HTTPException(400, "Hanya file PDF yang diterima")
            
        if pdf.size > 10 * 1024 * 1024:
            raise HTTPException(413, "Ukuran file melebihi 10MB")

        # Proses file PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            content = await pdf.read()
            if not content:
                raise HTTPException(400, "File PDF kosong")
            tmp_file.write(content)
            tmp_path = tmp_file.name

        try:
            # Gunakan PyMuPDF untuk menghindari warning
            loader = PyMuPDFLoader(tmp_path)
            raw_docs = loader.load()

            # Split dokumen
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=100,
                separators=["\n\n", "\n", ". ", " "]
            )
            
            split_docs = text_splitter.split_documents(raw_docs)
            
            # Cleaning dokumen
            cleaned_docs = []
            for doc in split_docs:
                try:
                    text = doc.page_content
                    text = re.sub(r'\s+', ' ', text).strip()
                    if text and len(text) > 50:  # Filter dokumen terlalu pendek
                        cleaned_docs.append(text)
                except Exception as e:
                    logger.warning(f"Gagal membersihkan dokumen: {str(e)}")
                    
            if not cleaned_docs:
                raise HTTPException(400, "Tidak ada teks yang valid dalam PDF")

            # Preprocessing
            processed_docs = [preprocess_text(doc) for doc in cleaned_docs]
            processed_docs = [doc for doc in processed_docs if doc]  # Filter empty lists
            
            if not processed_docs:
                raise HTTPException(400, "Gagal melakukan preprocessing teks")

            # Optimasi parameter model
            model_params = get_model_params(modelType)

            # Training model
            if modelType == 'word2vec':
                model = Word2Vec(processed_docs, **model_params)
            else:
                model = FastText(processed_docs, **model_params)

            # Penyimpanan hasil
            base_path = Path("model") / userId / chatbotId / pdfTitle
            storage_path = Path("storage") / userId / chatbotId / pdfTitle
            
            base_path.mkdir(parents=True, exist_ok=True)
            storage_path.mkdir(parents=True, exist_ok=True)

            # Simpan model
            model.save(str(base_path / f"{modelType}.model"))

            # Simpan metadata
            metadata = {
                "parameters": model_params,
                "statistik": {
                    "total_chunk": len(cleaned_docs),
                    "rata_panjang": sum(len(d) for d in cleaned_docs) // len(cleaned_docs),
                    "total_kata_unik": len(model.wv.key_to_index)
                }
            }
            
            with open(base_path / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

            # Simpan dokumen
            with open(storage_path / "original_texts.txt", "w") as f:
                f.write('\n'.join(cleaned_docs))

            # simpan preprocessed dokumen
            with open(storage_path / "preprocessed_texts.txt", "w") as f:
                for doc in processed_docs:
                    f.write(' '.join(doc) + '\n')
            
            logger.info(f"Training completed in {time.time() - start_time} seconds")

            return JSONResponse({
                "status": "sukses",
                "detail": metadata["statistik"],
                "path_model": str(base_path)
            }, status_code=201)

        except Exception as e:
            logger.error(f"Error processing: {str(e)}")
            raise HTTPException(500, "Gagal memproses PDF")

        finally:
            try:
                os.remove(tmp_path)
            except Exception as e:
                logger.error(f"Gagal menghapus file sementara: {str(e)}")

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(500, "Terjadi kesalahan internal")

@app.post("/query")
async def query_model(
    query: str = Form(...),
    userId: str = Form(...),
    chatbotId: str = Form(...),
    modelType: str = Form(..., regex="^(word2vec|fasttext)$"),
    pdfTitle: str = Form(...),
    topK: int = Form(default=5),
    similarityThreshold: float = Form(default=0.4),
):
    try:
        # Konstruksi path
        base_path = Path("model") / userId / chatbotId / pdfTitle
        storage_path = Path("storage") / userId / chatbotId / pdfTitle
        
        # 1. Load metadata
        try:
            with open(base_path / "metadata.json", "r") as f:
                metadata = json.load(f)
        except Exception as e:
            logger.error(f"Gagal memuat metadata: {str(e)}")
            raise HTTPException(400, "Metadata model tidak valid")

        # 2. Load model
        try:
            model_file = base_path / f"{modelType}.model"
            if modelType == 'word2vec':
                model = Word2Vec.load(str(model_file))
            else:
                model = FastText.load(str(model_file))
        except Exception as e:
            logger.error(f"Gagal memuat model: {str(e)}")
            raise HTTPException(404, "Model tidak ditemukan atau tidak valid")

        # 3. Load dokumen original
        try:
            text_file = storage_path / "original_texts.txt"
            with open(text_file, "r", encoding="utf-8") as f:
                documents = [line.strip() for line in f if line.strip()]
        except Exception as e:
            logger.error(f"Gagal memuat dokumen: {str(e)}")
            raise HTTPException(404, "Dokumen referensi tidak ditemukan")

        # Proses query
        results = find_context(
            query=query,
            embedding_model=model,
            docs=documents,
            top_k_max=topK,
            similarity_threshold=similarityThreshold
        )

        return JSONResponse({
            "status": "success",
            "results": results,
            "metadata": {
                "modelType": modelType,
                "parameters": metadata.get('parameters'),
                "documentsCount": len(documents)
            }
        }, status_code=200)
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(500, "Terjadi kesalahan internal")

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8888)