import re
import yaml
import numpy as np
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text.strip()

def create_sequences(
    texts,
    tokenizer=None,
    max_seq_len=config["model"]["max_seq_len"],
    min_context_len=config["training"]["min_context_len"],
    max_context_len=config["training"]["max_context_len"],
    vocab_size_limit=config["model"]["vocab_size_limit"],
    return_sentence_ids=False
):
    sequences = []
    sentence_ids = []


    for sid, t in enumerate(texts):
        tokens = word_tokenize(t)
        
        for i in range(min_context_len, len(tokens)):
            context_len = np.random.randint(min_context_len, min(max_context_len, i) + 1)
            sequences.append(tokens[i - context_len : i + 1])
            sentence_ids.append(sid)


    if tokenizer is None:
        tokenizer = Tokenizer(
            num_words=vocab_size_limit,
            oov_token="<OOV>"
        )
        tokenizer.fit_on_texts(sequences)

    sequences_int = tokenizer.texts_to_sequences(sequences)
    filtered = []
    filtered_sentence_ids = []

    for seq, sid in zip(sequences_int, sentence_ids): 
        if seq[-1] != tokenizer.word_index["<OOV>"]:
            filtered.append(seq)
            filtered_sentence_ids.append(sid)

    sequences_padded = pad_sequences(
        filtered, maxlen=max_seq_len, padding="pre"
    )
    X = sequences_padded[:, :-1]
    y = sequences_padded[:, -1]

    vocab_size = min(vocab_size_limit, len(tokenizer.word_index) + 1)
    if return_sentence_ids:
        return X, y, tokenizer, max_seq_len, vocab_size, filtered_sentence_ids

    return X, y, tokenizer, max_seq_len, vocab_size


def predict_next_word(
    model,
    tokenizer,
    text_seq,
    max_seq_len,
    k=5,
):  
    text_seq = preprocess_text(text_seq)
    token_list = tokenizer.texts_to_sequences([text_seq])[0]

    if not token_list:
        return ""
    token_list = pad_sequences(
        [token_list], maxlen=max_seq_len - 1, padding="pre"
    )
    probs = model.predict(token_list, verbose=0)[0]
    oov_index = tokenizer.word_index.get("<OOV>")
    if oov_index is not None:
        probs[oov_index] = 0.0

    frequent_words = {"the", "of", "to", "and", "in", "a"}
    for idx, word in tokenizer.index_word.items():
        if word in frequent_words:
            probs[idx] *= 0.2

    top_k_indices = probs.argsort()[-k:][::-1]

    top_k_probs = probs[top_k_indices]
    top_k_probs = top_k_probs / np.sum(top_k_probs)
    top_k_words = [(tokenizer.index_word.get(idx, ""), round(float(top_k_probs[i]), 4))
                   for i, idx in enumerate(top_k_indices)]

    return top_k_words

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
def get_word_embedding(word, tokenizer, embedding_matrix):
    idx = tokenizer.word_index.get(word)
    if idx is None or idx >= embedding_matrix.shape[0]:
        return None
    return embedding_matrix[idx]