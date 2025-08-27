import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
corpus = [
    "Artificial intelligence is transforming industries and society.",
    "Machine learning allows systems to improve from data.",
    "Deep learning is a subset of machine learning using neural networks.",
    "Climate change impacts weather patterns globally.",
    "Urban gardens help provide food security in cities.",
]
train_text = " ".join(corpus * 50)  
tokenizer = Tokenizer(oov_token='<OOV>')
tokenizer.fit_on_texts([train_text])
word_index = tokenizer.word_index
vocab_size = len(word_index) + 1
sequences = []
for line in train_text.split('. '):
    tokens = tokenizer.texts_to_sequences([line])[0]
    for i in range(2, len(tokens) + 1):
        seq = tokens[:i]
        sequences.append(seq)
max_len = max(len(s) for s in sequences)
sequences = pad_sequences(sequences, maxlen=max_len, padding='pre')
X = sequences[:, :-1]
y = sequences[:, -1]
model = Sequential([
    Embedding(vocab_size, 50, input_length=X.shape[1]),
    LSTM(128), 
    Dense(vocab_size, activation='softmax')
])
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
model.fit(X, y, epochs=40, batch_size=64, verbose=1)  
index_word = {v: k for k, v in tokenizer.word_index.items()}
def sample_with_top_k(preds, top_k=5, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds + 1e-9) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    # restrict to top-k
    top_indices = preds.argsort()[-top_k:]
    top_probs = preds[top_indices]
    top_probs = top_probs / np.sum(top_probs)
    return np.random.choice(top_indices, p=top_probs)
def generate_text(seed_text, gen_len=30, temperature=0.8, top_k=5):
    result = seed_text.split()
    used_words = set(result)  
    for _ in range(gen_len):
        token_list = tokenizer.texts_to_sequences([' '.join(result)])
        token_list = pad_sequences(token_list, maxlen=X.shape[1], padding='pre')
        preds = model.predict(token_list, verbose=0)[0]
        next_index = sample_with_top_k(preds, top_k=top_k, temperature=temperature)
        next_word = index_word.get(next_index, '')
        if not next_word:
            break
        if next_word in used_words:
            continue
        result.append(next_word)
        used_words.add(next_word)
        if next_word == '.':
            break
    return ' '.join(result)
print("Generated paragraph on AI:")
print(generate_text("Artificial intelligence", gen_len=40, temperature=0.8, top_k=5))
print("\nGenerated paragraph on Climate:")
print(generate_text("Climate change", gen_len=40, temperature=0.8, top_k=5))
