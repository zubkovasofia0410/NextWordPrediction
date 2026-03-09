import yaml
import nltk
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, GRU, LSTM
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from nltk.tokenize import sent_tokenize

from src.utils import preprocess_text, create_sequences
from src.reuters_extractor import extract_reuters_text
for res in ["punkt", "punkt_tab"]:
    try:
        nltk.data.find(f"tokenizers/{res}")
    except LookupError:
        nltk.download(res)

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
documents = extract_reuters_text(config["data"]["path"])

n_docs = len(documents)
test_size = int(config["training"]["test_split"] * n_docs)

train_docs = documents[:-test_size]
test_docs = documents[-test_size:]

train_sentences = [
    preprocess_text(s) 
    for doc in train_docs 
    for s in sent_tokenize(doc) 
    if len(s.split()) > 1
    ]

test_sentences = [
    preprocess_text(s) 
    for doc in test_docs 
    for s in sent_tokenize(doc) 
    if len(s.split()) > 1
    ]

train_sentences = train_sentences[:50000]
test_sentences = test_sentences[:500]

val_ratio = config["training"]["validation_split"]
n_val = int(len(train_sentences) * val_ratio)

val_sentences = train_sentences[-n_val:]
train_sentences = train_sentences[:-n_val]


X_train, y_train, tokenizer, max_seq_len, vocab_size = create_sequences(
    train_sentences,     
    max_seq_len=config["model"]["max_seq_len"],
    min_context_len=config["training"]["min_context_len"],
    max_context_len=config["training"]["max_context_len"],
    vocab_size_limit=config["model"]["vocab_size_limit"]
    )
X_val, y_val, _, _, _ = create_sequences(
    val_sentences, 
    tokenizer=tokenizer, 
    max_seq_len=max_seq_len,
    min_context_len=config["training"]["min_context_len"],
    max_context_len=config["training"]["max_context_len"],
    vocab_size_limit=config["model"]["vocab_size_limit"]
    )


model = Sequential([
    Embedding(
        input_dim=vocab_size,
        output_dim=config["model"]["embedding_dim"],
    ),
    LSTM(config["model"]["lstm_units"], dropout=0.3, recurrent_dropout=0.2),
    Dense(vocab_size, use_bias=False, activation="softmax")
])

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=config["training"]["early_stopping_patience"],
    restore_best_weights=True
)

model.fit(
    X_train,
    y_train,
    epochs=config["training"]["epochs"],
    batch_size=config["training"]["batch_size"],
    validation_data=(X_val, y_val),
    callbacks=[early_stop]
)

model.summary()

model.save(config["output"]["model_path"])

with open(config["output"]["tokenizer_path"], "wb") as f:
    pickle.dump(tokenizer, f)

with open(config["output"]["maxlen_path"], "w") as f:
    f.write(str(max_seq_len))

with open(config["output"]["test_sentences_path"], "w", encoding="utf-8") as f:
    for s in test_sentences:
        f.write(s + "\n")

