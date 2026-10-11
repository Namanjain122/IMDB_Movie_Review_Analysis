# IMDB_Movie_Review_Analysis
### 📘 README: Text Vectorization & Word Embeddings

This notebook demonstrates a modern Deep Learning pipeline for representing textual data as dense vector representations using **TensorFlow/Keras**.

---

### 🚀 Implemented Pipeline Breakdown

1. **Integer/Hash Encoding (`one_hot`)**
   * **Input:** Raw string sentences (e.g., `'the glass of milk'`).
   * **Action:** Uses a hashing function (`one_hot`) to deterministically map each word into a unique integer index constrained by a vocabulary size limit (`voc_size = 10000`). For example, `'glass'` gets mapped to `2704`.

2. **Sequence Padding (`pad_sequences`)**
   * **Action:** Standardizes input sizes. Sentences vary in length, but neural networks require uniform shape matrices.
   * By setting `sent_length = 8` and `padding='pre'`, shorter sentences are padded with leading zeros (`0`), resulting in an input array of shape `(7, 8)`.

3. **Dense Embedding Layer (`Embedding`)**
   * **Action:** Learns a dense projection space. It acts as a lookup dictionary of shape `(10000, 10)` containing trainable weights initialized randomly.
   * **Output:** Translates each of the `8` word indices into a continuous `10`-dimensional floating-point vector, producing an overall matrix shape of `(7, 8, 10)`.

---

### 📊 Conceptual Comparisons: How This Differs From Other Methods

| Vectorization Technique | Type of Representation | Size of Vector | Word Order Retained? | Captures Context/Semantic Meaning? | How it Works |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Implemented Keras Embedding** | **Dense** (Learned from scratch) | `dim = 10` (Low-dimensional) | **Yes** (Sequence structure is intact) | **Yes** (Learned jointly during model training) | Maps word indices to coordinates in a continuous multidimensional space; adjusts parameters using Backpropagation. |
| **One-Hot Encoding (OHE)** | **Sparse** (Static) | `voc_size = 10000` (High-dimensional) | **No** (When representing sentences) | **No** | Maps each word to a binary vector of size $V$ containing exactly one `1` and $V-1$ zeros. |
| **Bag of Words (BOW)** | **Sparse** (Static) | `voc_size = 10000` (Sentence aggregated) | **No** (Completely discarded) | **No** | Counts occurrences of words in a sentence; represents the document as a frequency histogram vector. |
| **N-Grams** | **Sparse** (Static) | Extremely High (Expands with vocabulary) | **Yes** (Local word sequences of size $N$) | **Minimal** (Only immediate neighbors) | Creates combinations of adjacent words (e.g., "the glass", "glass of") to preserve local ordering context. |
| **TF-IDF** | **Sparse** (Weighted) | `voc_size = 10000` | **No** | **No** | Modifies BOW frequency by penalizing words that appear too frequently across *all* documents (like "the", "is"). |
| **Word2Vec** | **Dense** (Pre-trained) | Typically `300` dimensions | **No** (Directly), **Yes** (In training context windows) | **Yes** (Highly semantic, pre-trained on billions of words) | An algorithm (CBOW or Skip-Gram) trained to predict words based on neighboring context, generating highly generalizable vectors. |
| **AvgWord2Vec** | **Dense** (Document aggregate) | Match dimensions (e.g., `300`) | **No** (Averages out the position) | **Yes** (Semantic concepts of words are retained) | Calculates the mathematical average (mean) of all individual pre-trained Word2Vec vectors present in a single sentence to output a uniform document-level embedding. |**
**
