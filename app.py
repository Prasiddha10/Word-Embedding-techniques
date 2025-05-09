from flask import Flask, request, jsonify, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import gensim.downloader as api
from gensim.models import Word2Vec
import nltk
from nltk.tokenize import word_tokenize
import os
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
app = Flask(__name__)

# Sample custom corpus
corpus = [
    "The quick brown fox jumps over the lazy dog",
    "Never jump over lazy dogs quickly",
    "Fast foxes and sleepy dogs",
    "Quick brown foxes leap over sleeping hounds",
    "Dogs and foxes are canines"
]

# TF-IDF setup
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)

# GloVe setup (using pre-trained model)
try:
    glove_vectors = api.load("glove-wiki-gigaword-100")
except:
    print("Downloading GloVe model...")
    glove_vectors = api.load("glove-wiki-gigaword-100")

# Train Word2Vec on our corpus for comparison
tokenized_corpus = [word_tokenize(doc.lower()) for doc in corpus]
word2vec_model = Word2Vec(tokenized_corpus, vector_size=100, window=5, min_count=1, workers=4)

@app.route('/')
def index():
    return render_template('embeddings.html')

@app.route('/api/embeddings', methods=['POST'])
def get_embeddings():
    data = request.json
    words = data.get('words', [])
    method = data.get('method', 'glove')
    
    results = []
    for word in words:
        try:
            if method == 'tfidf':
                # For TF-IDF, we use the average of document vectors containing the word
                word_indices = [i for i, doc in enumerate(corpus) if word.lower() in doc.lower()]
                if word_indices:
                    vector = tfidf_matrix[word_indices].mean(axis=0).A1
                else:
                    vector = np.zeros(tfidf_matrix.shape[1])
            elif method == 'glove':
                vector = glove_vectors[word.lower()] if word.lower() in glove_vectors else np.zeros(100)
            elif method == 'word2vec':
                vector = word2vec_model.wv[word.lower()] if word.lower() in word2vec_model.wv else np.zeros(100)
            else:
                return jsonify({"error": "Invalid method"}), 400
            
            results.append({
                "word": word,
                "vector": vector.tolist(),
                "method": method
            })
        except Exception as e:
            results.append({
                "word": word,
                "error": str(e)
            })
    
    return jsonify({"results": results})

@app.route('/api/visualize', methods=['POST'])
def visualize_embeddings():
    data = request.json
    words = data.get('words', [])
    method = data.get('method', 'glove')
    reduction = data.get('reduction', 'tsne')
    
    # Get vectors
    vectors = []
    valid_words = []
    for word in words:
        try:
            if method == 'tfidf':
                word_indices = [i for i, doc in enumerate(corpus) if word.lower() in doc.lower()]
                if word_indices:
                    vector = tfidf_matrix[word_indices].mean(axis=0).A1
                else:
                    continue
            elif method == 'glove':
                vector = glove_vectors[word.lower()] if word.lower() in glove_vectors else None
            elif method == 'word2vec':
                vector = word2vec_model.wv[word.lower()] if word.lower() in word2vec_model.wv else None
            else:
                return jsonify({"error": "Invalid method"}), 400
            
            if vector is not None:
                vectors.append(vector)
                valid_words.append(word)
        except:
            continue
    
    if not vectors:
        return jsonify({"error": "No valid words found"}), 400
    
    vectors = np.array(vectors)
    
    # Dimensionality reduction
    if reduction == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
    else:
        reducer = PCA(n_components=2)
    
    reduced_vectors = reducer.fit_transform(vectors)
    
    # Create plot
    plt.figure(figsize=(10, 8))
    plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1])
    for i, word in enumerate(valid_words):
        plt.annotate(word, (reduced_vectors[i, 0], reduced_vectors[i, 1]))
    
    plt.title(f"{method.upper()} Embeddings ({reduction.upper()} Reduction)")
    
    # Save plot to bytes
    img_bytes = BytesIO()
    plt.savefig(img_bytes, format='png')
    img_bytes.seek(0)
    plt.close()
    
    return jsonify({
        "image": base64.b64encode(img_bytes.read()).decode('utf-8'),
        "coordinates": {word: coords.tolist() for word, coords in zip(valid_words, reduced_vectors)}
    })

@app.route('/api/neighbors', methods=['POST'])
def get_neighbors():
    data = request.json
    word = data.get('word', '')
    method = data.get('method', 'glove')
    topn = data.get('topn', 5)
    
    try:
        if method == 'glove':
            if word.lower() not in glove_vectors:
                return jsonify({"error": "Word not in vocabulary"}), 404
            neighbors = glove_vectors.most_similar(word.lower(), topn=topn)
        elif method == 'word2vec':
            if word.lower() not in word2vec_model.wv:
                return jsonify({"error": "Word not in vocabulary"}), 404
            neighbors = word2vec_model.wv.most_similar(word.lower(), topn=topn)
        else:
            return jsonify({"error": "Method not supported for neighbors"}), 400
        
        return jsonify({
            "word": word,
            "neighbors": [{"word": w, "similarity": float(s)} for w, s in neighbors],
            "method": method
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)