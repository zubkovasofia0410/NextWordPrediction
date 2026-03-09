# Next-Word Prediction with LSTM on Reuters-21578

This project implements a **next-word prediction language model** trained on the **Reuters-21578 corpus**.  
The model learns from existing word sequences and predicts the most likely next word given a short context.

The goal is **sequence modeling**, with evaluation performed on *held-out, unseen sentences*.

---

## Project Structure

```

│
├── config.yaml # Central configuration file
├── requirements.txt # Python dependencies
│
├── data/
│ └── reut2-0**.sgm # Reuters SGML files
│
├── src/
│ ├── utils.py # Text preprocessing & sequence generation
│ ├── __init__.py 
│ └── reuters_extractor.py # SGML parsing for Reuters corpus
│ └── plot_model.py # script to generate and save the model architecture as a figure
│
├── experiments/
│ ├── train_model.py # Train and save the language model
│ └── test_model.py # Evaluate, generate predictions and generate plot and metrics
│
├── models/
│ ├── saved_model.keras # Saved trained model
│ ├── tokenizer.pkl # Saved tokenizer
│ ├── max_seq_len.txt # Sequence length used in training
│ └── test_sentences.txt # Held-out sentences for evaluation
│
└── README.md
```

---

## Setup: Install dependencies and dataset preparation

``` python
pip install -r requirements.txt
```
* NLTK resources (punkt, punkt_tab) are downloaded automatically on first run.


Update config.yaml if needed:

``` yaml
data:
  path: data/reut2-000.sgm
``` 

* You may use a single .sgm file for faster experimentation or multiple files for larger training.

## Training the Model

Run training from the project root:

``` python
python -m experiments.train_model
```

This will:

* Parse Reuters SGML documents

* Split documents into sentences

* Hold out unseen sentences for testing

* Train an LSTM-based language model

* Save the trained model, tokenizer, maximum sequence length, and test sentences

Training progress and validation loss will be printed to the console.

## Testing & Evaluation (No Retraining)

After training completes, you can run:

```python
python -m experiments.test_model
```

This will:

* Load the saved model and tokenizer

* Evaluate Top-1 and Top-5 accuracy on held-out test sentences

* Generate qualitative next-word   predictions, including:

    * Randomly sampled contexts with variable lengths

    * Top-1 and Top-5 predicted words with probabilities

    * Cosine similarity of predicted words to the true next word

    * Top 5 sentences with BEST accuracy

    * Top 5 sentences with WORST accuracy

    * Top 5 LONGEST sentences

## Configuration (config.yaml)

Key parameters you may tune:

* model:
    - embedding_dim: 100           # Dimension of word embeddings
    - lstm_units: 64            # Number of LSTM hidden units
    - max_seq_len: 50           # Maximum input sequence length
    - vocab_size_limit: 6000    # Maximum vocabulary size

* training:
    - epochs: 20
    - batch_size: 128
    - validation_split: 0.1
    - test_split: 0.2
    - min_context_len: 4
    - max_context_len: 10
    - early_stopping_patience: 3

* output:
    - model_path: models/saved_model.keras
    - tokenizer_path: models/tokenizer.pkl
    - maxlen_path: models/max_seq_len.txt
    - test_sentences_path: models/test_sentences.txt

Adjust embedding_dim, lstm_units, and max_seq_len to trade off model capacity and training time.
The min_context_len and max_context_len control variable-length input sequences for prediction.