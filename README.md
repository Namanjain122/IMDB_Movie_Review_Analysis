# IMDB Movie Review Analysis


https://github.com/user-attachments/assets/c60ce0e8-9e66-4f9e-b46e-ecfb4b4648ad



https://github.com/user-attachments/assets/1f1bd009-585b-4d6d-ab8d-08056e4f425b




Live Link: https://imdbmoviereviewanalysisgit-doiaek6gdnvjdnw3yssuhk.streamlit.app/

An end-to-end sentiment-analysis project using TensorFlow/Keras and a Simple
Recurrent Neural Network (SimpleRNN). The project includes notebooks for
exploring the IMDB dataset, training and using the model, and learning how
integer sequences are converted into dense word embeddings. A Streamlit app
provides an interface for classifying a movie review as positive or negative.

## Project contents

| Path | Purpose |
| --- | --- |
| `Main.py` | Streamlit application for interactive sentiment prediction. |
| `Models/simple_rnn_imdb.h5` | Model file loaded by the Streamlit app. |
| `Models/SimpleRNN.h5` | Additional saved SimpleRNN model artifact. |
| `Trained_SIMPLE_RNN_IMDB.ipynb` | IMDB data preparation, model training, and model saving. |
| `RNN_Implementation_IMDB_dataset_Feature_Engineering.ipynb` | IMDB dataset loading, review decoding, and sequence preparation. |
| `RNN_Implementation_IMDB_Prediction.ipynb` | Loading the saved model and making a prediction from review text. |
| `embeddings.ipynb` | Demonstration of integer encoding, padding, and Keras embeddings. |
| `requirements.txt` | Python packages used by the project. |

## How the sentiment model works

The IMDB dataset provided by Keras contains 25,000 training reviews and 25,000
test reviews, with binary sentiment labels. The notebooks load up to 10,000
word indices and pad or truncate reviews to a fixed length of 500 tokens.

The model in `Trained_SIMPLE_RNN_IMDB.ipynb` uses this architecture:

1. An `Embedding` layer maps each token index to a trainable 128-dimensional
   vector.
2. A `SimpleRNN` layer with 128 units processes the sequence.
3. A one-unit `Dense` layer with sigmoid activation outputs a score between 0
   and 1.

The model is compiled with Adam and binary cross-entropy. Training uses batches
of 32, up to 10 epochs, a 20% validation split, and early stopping on validation
loss (patience 5, restoring the best weights).

The app treats a score greater than 0.5 as **Positive** and all other scores as
**Negative**. The displayed score is the model's sigmoid output, not a
guaranteed measure of certainty.

## Setup

Use a Python environment compatible with the TensorFlow version available for
your platform. From the project directory, create and activate a virtual
environment, then install the listed dependencies.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

TensorFlow downloads the IMDB dataset the first time a notebook or the app
requests it. An internet connection may be required for this initial download.

## Run the Streamlit app

From the project directory, run:

```powershell
streamlit run Main.py
```

Open the local URL printed by Streamlit, enter a review in the text area, and
select **classify**. The app loads `Models/simple_rnn_imdb.h5` relative to the
current working directory, so launch Streamlit from the project directory and
keep that model file at the expected path.

## Train the model

Open `Trained_SIMPLE_RNN_IMDB.ipynb` in Jupyter or VS Code and execute its cells
in order. It downloads and prepares the dataset, defines and trains the model,
then saves the resulting model as `simple_rnn_imdb.h5` in the notebook's
working directory.

The Streamlit app instead loads `Models/simple_rnn_imdb.h5`. To use a newly
trained model in the app, place the saved model at that path (or update the
model path in `Main.py`). Keep the existing model artifact if you want to
continue using the currently supplied model.

`RNN_Implementation_IMDB_dataset_Feature_Engineering.ipynb` walks through
loading the IMDB data, decoding an indexed review, and padding sequences.
`RNN_Implementation_IMDB_Prediction.ipynb` demonstrates loading the saved
model and predicting the sentiment of a sample review.

## Text vectorization and word embeddings

`embeddings.ipynb` demonstrates a small text-vectorization pipeline using
TensorFlow/Keras. It is separate from the IMDB model's data preparation and
serves as an educational example of how text can be represented for a neural
network.

### Demonstration pipeline

1. **Integer/hash encoding with `one_hot`**

   The notebook passes seven example sentences to Keras'
   `tensorflow.keras.preprocessing.text.one_hot` with `voc_size = 10000`.
   The function maps words to integer IDs in the configured range. This is a
   hashing-based mapping, so IDs are not guaranteed to be unique for distinct
   words; collisions are possible.

2. **Sequence padding with `pad_sequences`**

   Sentences have different lengths, while a batch of model inputs needs a
   consistent shape. With `sent_length = 8` and `padding='pre'`, shorter
   sequences are padded with zeros at the beginning and longer sequences are
   truncated to the maximum length. For the seven sample sentences, the padded
   input has shape `(7, 8)`.

3. **Dense vectors with `Embedding`**

   The notebook creates an `Embedding` layer with `input_dim = 10000` and
   `output_dim = 10`. Each input token index selects a trainable 10-dimensional
   vector. Passing the padded batch through the layer produces a tensor of
   shape `(7, 8, 10)`: seven sentences, eight token positions per sentence,
   and ten values per token.

   The embedding weights are initialized by the framework and are trainable.
   In the notebook they are inspected after a forward pass; the demonstration
   does not train them on a language or sentiment objective, so the initial
   vectors should not be interpreted as learned semantic representations.

### Conceptual comparison of text representations

| Technique | Representation | Typical size | Preserves word order? | Semantic/context information | How it represents text |
| --- | --- | --- | --- | --- | --- |
| **Keras `Embedding` (this notebook)** | Dense, trainable | `10` dimensions per token in the example | Yes, when vectors remain in sequence order | Can learn task-specific patterns when trained with a task objective | Looks up a trainable vector for each token ID. |
| **One-hot encoding** | Sparse, static | Vocabulary size (`10000` in the example) | Not by itself; each token is represented independently | No inherent semantic similarity | Represents a token with one active position and zeros elsewhere. |
| **Bag of Words (BoW)** | Sparse, static | Vocabulary size | No; aggregates token counts | No inherent semantic similarity | Counts vocabulary terms in a document. |
| **N-grams** | Usually sparse, static | Often much larger than the vocabulary | Preserves local sequences up to the chosen n-gram length | Limited; captures local co-occurrence, not general meaning | Represents adjacent word groups such as “the glass” or “glass of”. |
| **TF-IDF** | Sparse, weighted | Vocabulary size | No, unless combined with n-grams | No inherent semantic similarity | Weights term counts by how distinctive terms are across documents. |
| **Word2Vec** | Dense, commonly pre-trained | Often 300 dimensions, depending on the model | Not as a document sequence representation; training uses context windows | Captures distributional word similarities learned from its training corpus | Learns word vectors with CBOW or Skip-gram context-prediction objectives. |
| **Average Word2Vec** | Dense, document-level aggregate | Same dimension as its word vectors | No; averaging removes token positions | Retains some word-level distributional signal | Averages available word vectors to make one vector for a document. |

One-hot vectors, BoW, and TF-IDF are not learned dense embeddings. Word2Vec
vectors are learned from a corpus and can be used as pre-trained features.
Likewise, a Keras embedding layer only learns useful semantic or
task-specific structure when its weights are trained on a suitable objective.

## Notes and limitations

- `Main.py` tokenizes user input with lowercase whitespace splitting and maps
  words through the Keras IMDB word index. Punctuation is not explicitly
  normalized, so user-entered reviews may tokenize differently from the
  dataset's preprocessing.
- The app pads or truncates each review to 500 tokens, matching the notebooks'
  sequence length.
- `requirements.txt` does not pin package versions. If an install or model
  loading error occurs, check TensorFlow/Keras compatibility with the saved
  `.h5` artifact and your Python version.
- The saved `.h5` files are required model artifacts for inference; the
  application does not train a model when it starts.
