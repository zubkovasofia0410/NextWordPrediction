import yaml
import pickle
import nltk
import numpy as np
from tensorflow.keras.models import load_model
from src.utils import create_sequences
import random
import matplotlib.pyplot as plt
from collections import defaultdict
from src.utils import predict_next_word, get_word_embedding, cosine_similarity


for res in ["punkt", "punkt_tab"]:
    try:
        nltk.data.find(f"tokenizers/{res}")
    except LookupError:
        nltk.download(res)

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

model = load_model(config["output"]["model_path"])

with open(config["output"]["tokenizer_path"], "rb") as f:
    tokenizer = pickle.load(f)

with open(config["output"]["maxlen_path"], "r") as f:
    max_seq_len = int(f.read())

with open(config["output"]["test_sentences_path"], "r", encoding="utf-8") as f:
    test_sentences = [line.strip() for line in f if line.strip()]

min_context_len = config["training"]["min_context_len"]
max_context_len = config["training"]["max_context_len"]
vocab_size_limit = config["model"]["vocab_size_limit"]

X_test, y_test, _, _, _, sentence_ids = create_sequences(
    test_sentences,
    tokenizer=tokenizer,
    min_context_len = min_context_len,
    max_context_len = max_context_len,
    vocab_size_limit = vocab_size_limit,
    return_sentence_ids=True
)
# how accurate is model, verbose=1 says to show results in terminal
loss, acc = model.evaluate(X_test, y_test, verbose=1)
print(f"Top-1 word Test accuracy (unseen sentences, no retraining): {acc:.4f}")

pred_probs = model.predict(X_test, verbose=0)
top1_preds = np.argmax(pred_probs, axis=1)
topk_preds = np.argsort(pred_probs, axis=1)[:, -5:][:, ::-1]
topk_correct = np.array([y_test[i] in topk_preds[i] for i in range(len(y_test))])
topk_acc = np.mean(topk_correct)
print(f"Top-5 words accuracy: {topk_acc:.4f}")

embedding_matrix = model.layers[0].get_weights()[0]


def qualitative_examples(
    model,
    tokenizer,
    sentences,
    max_seq_len,
    embedding_matrix,
    n_examples=20
):
    print("\nQualitative next-word predictions:\n")

    for _ in range(n_examples):
        sent = random.choice(sentences)
        tokens = sent.split()

        context_len = random.randint(min_context_len, max_context_len)

        if len(tokens) <= context_len:
            continue

        context = " ".join(tokens[:context_len])
        true_next = tokens[context_len]

        top_k = predict_next_word(
            model, tokenizer, context, max_seq_len, k=5
        )

        true_vec = get_word_embedding(
            true_next, tokenizer, embedding_matrix
        )

        print(f"Context length: {context_len}")
        print(f"Input context : '{context}'")
        print(f"True next word: '{true_next}'")
        print("Top predictions:")

        for word, prob in top_k:
            pred_vec = get_word_embedding(
                word, tokenizer, embedding_matrix
            )

            if true_vec is not None and pred_vec is not None:
                sim = cosine_similarity(true_vec, pred_vec)
                sim_str = f"{sim:.3f}"
            else:
                sim_str = "N/A"

            print(f"  {word:15s} prob={prob:.4f}  sim={sim_str}")

        print("-" * 60)



sentence_stats = {}

for i, sid in enumerate(sentence_ids):

    if sid not in sentence_stats:
        sentence_stats[sid] = {
            "correct_top1": 0,
            "correct_topk": 0,
            "total": 0,
            "examples": []
        }

    stats = sentence_stats[sid]
    stats["total"] += 1

    if top1_preds[i] == y_test[i]:
        stats["correct_top1"] += 1

    if y_test[i] in topk_preds[i]:
        stats["correct_topk"] += 1
    true_word = tokenizer.index_word[y_test[i]]
    pred_word = tokenizer.index_word[top1_preds[i]]
    topk_words = [tokenizer.index_word[idx] for idx in topk_preds[i]]

    stats["examples"] = [{
    "true": true_word,
    "pred": pred_word,
    "topk": topk_words
}]

sentence_scores = []
for sid, stats in sentence_stats.items():
    sentence_scores.append({
        "sentence": test_sentences[sid],
        "top1_acc": stats["correct_top1"] / stats["total"],
        "topk_acc": stats["correct_topk"] / stats["total"],
        "n_preds": stats["total"],
        "examples": stats["examples"]

    })
longest = sorted(
    sentence_scores,
    key=lambda x: x["n_preds"],
    reverse=True
)

best = sorted(sentence_scores, key=lambda x: x["topk_acc"], reverse=True)
worst = sorted(sentence_scores, key=lambda x: x["topk_acc"])

def print_sentence_rankings(sentences, title, n=5):
    print(f"\n{title}\n" + "=" * len(title))
    for s in sentences[:n]:
        print(f"\nTop-1 acc: {s['top1_acc']:.3f}")
        print(f"Top-k acc: {s['topk_acc']:.3f}")
        print(f"Predictions: {s['n_preds']}")
        print(f"Sentence: {s['sentence']}")
        print("Examples:")

        for ex in s["examples"]:
            print(
                f"  true: '{ex['true']}' | pred: '{ex['pred']}' "
                f"| top-k: {ex['topk']}"
            )


qualitative_examples(
    model,
    tokenizer,
    test_sentences,
    max_seq_len,
    embedding_matrix,
    n_examples=10
    )

print_sentence_rankings(best, "BEST sentences (Top-k accuracy)", n=5)
print_sentence_rankings(worst, "WORST sentences (Top-k accuracy)", n=5)
print_sentence_rankings(longest, "LONGEST sentences (Most prediction steps)", n=5)


context_stats = defaultdict(lambda: {
    "total": 0,
    "correct_top1": 0,
    "correct_topk": 0
})


for i in range(len(X_test)):
    context_len = np.count_nonzero(X_test[i])

    context_stats[context_len]["total"] += 1

    if top1_preds[i] == y_test[i]:
        context_stats[context_len]["correct_top1"] += 1

    if y_test[i] in topk_preds[i]:
        context_stats[context_len]["correct_topk"] += 1

context_lengths = sorted(context_stats.keys())

top1_acc_ctx = [
    context_stats[l]["correct_top1"] / context_stats[l]["total"]
    for l in context_lengths
]

topk_acc_ctx = [
    context_stats[l]["correct_topk"] / context_stats[l]["total"]
    for l in context_lengths
]

plt.figure()
plt.plot(context_lengths, top1_acc_ctx, marker="o")
plt.plot(context_lengths, topk_acc_ctx, marker="o")
plt.xlabel("Context length (number of input words)")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Context Length")
plt.legend(["Top-1 accuracy", "Top-5 accuracy"])
plt.tight_layout()
plt.savefig("img/accuracy_vs_context_length.png")
plt.close()


acc_gap = [
    topk_acc_ctx[i] - top1_acc_ctx[i]
    for i in range(len(context_lengths))
]

plt.figure()
plt.plot(context_lengths, acc_gap, marker="o")
plt.xlabel("Context length")
plt.ylabel("Top-5 − Top-1 accuracy")
plt.title("Prediction Ambiguity vs Context Length")
plt.tight_layout()
plt.savefig("img/accuracy_gap_vs_context_length.png")
plt.close()


sims_top1_correct = []
sims_top1_incorrect = []
sims_topk = []

k = 5 

for i in range(len(y_test)):
    true_word = tokenizer.index_word.get(y_test[i])
    if true_word is None:
        continue
    true_vec = get_word_embedding(true_word, tokenizer, embedding_matrix)
    if true_vec is None:
        continue

    pred_word = tokenizer.index_word.get(top1_preds[i])
    if pred_word is None:
        continue
    pred_vec = get_word_embedding(pred_word, tokenizer, embedding_matrix)
    if pred_vec is None:
        continue

    sim_top1 = cosine_similarity(true_vec, pred_vec)

    if top1_preds[i] == y_test[i]:
        sims_top1_correct.append(sim_top1)
    else:
        sims_top1_incorrect.append(sim_top1)

    max_sim_k = -1
    for idx in topk_preds[i]:
        pred_word_k = tokenizer.index_word.get(idx)
        if pred_word_k is None:
            continue
        pred_vec_k = get_word_embedding(pred_word_k, tokenizer, embedding_matrix)
        if pred_vec_k is None:
            continue
        sim_k = cosine_similarity(true_vec, pred_vec_k)
        if sim_k > max_sim_k:
            max_sim_k = sim_k
    sims_topk.append(max_sim_k)

plt.figure(figsize=(8,5))
plt.boxplot(
    [sims_top1_correct, sims_top1_incorrect, sims_topk],
    labels=["Top-1 Correct", "Top-1 Incorrect", "Top-5 Max Similarity"],
    notch=True,
    showfliers=False
)
plt.ylabel("Cosine similarity")
plt.title("Semantic similarity of predictions")
plt.tight_layout()
plt.savefig("img/cosine_similarity_topk.png")
plt.close()

print("Boxplot statistics:")
for name, sims in zip(
    ["Top-1 Correct", "Top-1 Incorrect", "Top-5 Max Similarity"],
    [sims_top1_correct, sims_top1_incorrect, sims_topk]
):
    print(f"{name}: mean = {np.mean(sims):.3f}, median = {np.median(sims):.3f}")


word_counts = tokenizer.word_counts

freq_stats = defaultdict(lambda: {
    "total": 0,
    "correct": 0
})

for i in range(len(y_test)):
    true_word = tokenizer.index_word.get(y_test[i])
    if true_word is None:
        continue

    freq = word_counts.get(true_word, 0)

    freq_stats[freq]["total"] += 1
    if top1_preds[i] == y_test[i]:
        freq_stats[freq]["correct"] += 1

freqs = sorted(freq_stats.keys())
acc_by_freq = [
    freq_stats[f]["correct"] / freq_stats[f]["total"]
    for f in freqs
]

plt.figure()
plt.plot(freqs, acc_by_freq, marker="o")
plt.xscale("log")
plt.xlabel("Word frequency (log scale)")
plt.ylabel("Top-1 accuracy")
plt.title("Accuracy vs True Word Frequency")
plt.tight_layout()
plt.savefig("img/accuracy_vs_word_frequency.png")
plt.close()


