from flask import Flask, request, jsonify
import os
import gensim
import numpy as np
from gensim.models import Word2Vec, FastText
import tempfile
import re
import nltk
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.tokenize import RegexpTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Inisialisasi Flask App
app = Flask(__name__)

# NLTK Setup
nltk.download('punkt')
nltk.download('stopwords')

# Konfigurasi
EMBEDDING_DIM = 300

# Inisialisasi stemmer dan tokenizer
factory = StemmerFactory()
stemmer = factory.create_stemmer()
tokenizer = RegexpTokenizer(r'\w+')

# Preprocessing teks untuk bahasa Indonesia
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = stemmer.stem(text)
    
    tokens = tokenizer.tokenize(text)
    
    stop_words = stopwords.words('indonesian')
    custom_stopwords = {
        'yg', 'dg', 'dgn', 'ny', 'sih', 'nya', 'kalo', 'deh', 'mah',
        'lah', 'dll', 'tsb', 'dr', 'pd', 'utk', 'sd', 'dpt', 'dlm',
        'thn', 'tgl', 'jd', 'tkr', 'org', 'sbg', 'bs', 'tsb', 'kpd'
    }
    stop_words = set(stop_words).union(custom_stopwords)
    
    return [word for word in tokens if (
        word not in stop_words and
        len(word) > 2 and
        not any(char.isdigit() for char in word)
    )]

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
    
    return [docs[i] for i in top_indices]

@app.route('/train', methods=['POST'])
def train_model():
    try:
        # Validasi file PDF
        if 'pdf' not in request.files:
            return jsonify({'error': 'No PDF file uploaded'}), 400
            
        pdf_file = request.files['pdf']
        if pdf_file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
            
        # Validasi form data
        data = request.form
        required_fields = ['userId', 'chatbotId', 'modelType', 'pdfTitle']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'}), 400

        # Ekstrak parameter
        user_id = str(data['userId'])
        chatbot_id = str(data['chatbotId'])
        model_type = data['modelType']
        pdf_title = data['pdfTitle']

        # Simpan PDF sementara
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            pdf_file.save(tmp_file.name)
            tmp_path = tmp_file.name

        try:
            # Proses PDF dengan LangChain
            loader = PyPDFLoader(tmp_path)
            raw_documents = loader.load()

            # Split dokumen
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=150,
                length_function=len,
                separators=["\n\n", "\n", "(?<=\. )", " ", ""]
            )
            
            split_documents = text_splitter.split_documents(raw_documents)
            documents = [doc.page_content for doc in split_documents]

            # Cleaning tambahan
            cleaned_documents = []
            for doc in documents:
                text = doc.replace('\n', ' ').replace('\t', ' ')
                text = re.sub(r'\s+', ' ', text).strip()
                if text:
                    cleaned_documents.append(text)

            # Preprocessing
            processed_docs = [preprocess_text(doc) for doc in cleaned_documents]

            # Training model
            if model_type == 'word2vec':
                model = Word2Vec(processed_docs, vector_size=EMBEDDING_DIM, window=5, min_count=1, workers=4)
            elif model_type == 'fasttext':
                model = FastText(processed_docs, vector_size=EMBEDDING_DIM, window=5, min_count=1, workers=4)
            else:
                return jsonify({'error': 'Invalid model type'}), 400

            # Path penyimpanan
            base_model_path = os.path.join('model', user_id, chatbot_id, pdf_title)
            base_storage_path = os.path.join('storage', user_id, chatbot_id, pdf_title)
            os.makedirs(base_model_path, exist_ok=True)
            os.makedirs(base_storage_path, exist_ok=True)

            # Simpan model
            model.save(os.path.join(base_model_path, f'{model_type}.model'))
            with open(os.path.join(base_model_path, 'model_type.txt'), 'w') as f:
                f.write(model_type)

            # Simpan dokumen
            with open(os.path.join(base_storage_path, 'original_texts.txt'), 'w') as f:
                f.write('\n'.join(cleaned_documents))
                
            with open(os.path.join(base_storage_path, 'preprocessedText.txt'), 'w') as f:
                for doc in processed_docs:
                    f.write(' '.join(doc) + '\n')

            return jsonify({
                'message': 'Model trained successfully',
                'stats': {
                    'total_chunks': len(cleaned_documents),
                    'average_length': sum(len(d) for d in cleaned_documents)//len(cleaned_documents)
                }
            }), 201

        finally:
            # Bersihkan file temporer
            os.remove(tmp_path)

    except Exception as e:
        return jsonify({'error': str(e)}), 500
        
@app.route('/query', methods=['POST'])
def query():
    try:
        data = request.get_json()
        
        # Validasi input
        required_fields = ['query', 'embeddingModel', 'userId', 'chatbotId', 'pdfTitle', 'top_k_max', 'similarity_threshold']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'}), 400
                
        # Ekstrak parameter
        embedding_model_path = data['embeddingModel']
        user_id = str(data['userId'])
        chatbot_id = str(data['chatbotId'])
        pdf_title = data['pdfTitle']
        
        # Load model
        model_dir = os.path.dirname(embedding_model_path)
        with open(os.path.join(model_dir, 'model_type.txt'), 'r') as f:
            model_type = f.read().strip()
            
        if model_type == 'word2vec':
            model = Word2Vec.load(embedding_model_path)
        else:
            model = FastText.load(embedding_model_path)
        
        # Load original texts
        storage_path = os.path.join('storage', user_id, chatbot_id, pdf_title, 'original_texts.txt')
        with open(storage_path, 'r') as f:
            documents = [line.strip() for line in f.readlines()]
        
        # Proses query
        results = find_context(
            query=data['query'],
            embedding_model=model,
            docs=documents,
            top_k_max=data['top_k_max'],
            similarity_threshold=data['similarity_threshold']
        )
        
        return jsonify(results), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6666)