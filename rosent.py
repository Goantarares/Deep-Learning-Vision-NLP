import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding, SimpleRNN, LSTM, Bidirectional,
    Dense, Dropout, BatchNormalization, Conv1D, MaxPooling1D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import tensorflow as tf
import pandas as pd
from collections import Counter
import re
from sklearn.metrics import confusion_matrix, classification_report

# Configurare reproducibilitate
np.random.seed(42)
tf.random.set_seed(42)

# Detectare mediu Kaggle vs local
if os.path.exists('/kaggle/input'):
    BASE_PATH = '/kaggle/input/ro-sent'
    OUTPUT_DIR = '/kaggle/working/explorare_ro_sent'
    RESULTS_DIR = '/kaggle/working/rezultate_rnn'
    print("Mediu Kaggle")
else:
    BASE_PATH = 'ro_sent'
    OUTPUT_DIR = 'explorare_ro_sent'
    RESULTS_DIR = 'rezultate_rnn'
    print("Mediu local")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# INCARCARE DATE


train_df = pd.read_csv(f'{BASE_PATH}/train.csv', index_col=0)
test_df = pd.read_csv(f'{BASE_PATH}/test.csv', index_col=0)

train_df = train_df.dropna(subset=['text'])
test_df = test_df.dropna(subset=['text'])

print(f"Train: {len(train_df)} exemple")
print(f"Test: {len(test_df)} exemple")

label_map = {0: 'Negativ', 1: 'Pozitiv'}
train_df['sentiment'] = train_df['label'].map(label_map)
test_df['sentiment'] = test_df['label'].map(label_map)

# ECHILIBRUL CLASELOR


train_dist = train_df['sentiment'].value_counts()
ratio = max(train_dist.values) / min(train_dist.values)
print(f"Ratio dezechilibru: {ratio:.2f}:1")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.countplot(x='sentiment', data=train_df, ax=axes[0], palette='viridis')
axes[0].set_title('Echilibru Clase - Train', fontsize=14, fontweight='bold')
for container in axes[0].containers:
    axes[0].bar_label(container)

sns.countplot(x='sentiment', data=test_df, ax=axes[1], palette='magma')
axes[1].set_title('Echilibru Clase - Test', fontsize=14, fontweight='bold')
for container in axes[1].containers:
    axes[1].bar_label(container)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/01_echilibru_clase.png", dpi=300, bbox_inches='tight')
plt.close()


# STATISTICI TEXT

train_df['word_count'] = train_df['text'].apply(lambda x: len(x.split()))
train_df['char_length'] = train_df['text'].apply(len)
test_df['word_count'] = test_df['text'].apply(lambda x: len(x.split()))
test_df['char_length'] = test_df['text'].apply(len)

print(f"Lungime medie (cuvinte): {train_df['word_count'].mean():.1f}")
print(f"Lungime medie (caractere): {train_df['char_length'].mean():.1f}")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sns.histplot(data=train_df, x='word_count', hue='sentiment', kde=True, bins=50, ax=axes[0], palette=['red', 'green'])
axes[0].set_title('Distribuție Lungimi (Cuvinte)', fontsize=14, fontweight='bold')
axes[0].set_xlim(0, 400)

sns.boxplot(data=train_df, x='sentiment', y='word_count', ax=axes[1], palette=['red', 'green'])
axes[1].set_title('Comparație Lungimi (Boxplot)', fontsize=14, fontweight='bold')
axes[1].set_ylim(0, 400)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/02_distributie_cuvinte.png", dpi=300, bbox_inches='tight')
plt.close()


# CELE MAI FRECVENTE CUVINTE

STOP_WORDS = set([
    "și", "si", "sau", "dar", "că", "ca", "de", "la", "în", "in", "pe", "cu",
    "din", "pentru", "este", "sunt", "fi", "eu", "tu", "el", "ea", "mai", "nu",
    "se", "le", "un", "ce", "care", "cum", "când", "cand", "unde", "am", "ai",
    "are", "au", "tot", "foarte", "mult", "cel", "cea", "prin", "până", "pana",
    "după", "dupa", "între", "intre", "fără", "fara", "despre"
])


def clean_text(text):
    text = text.lower()
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'http\S+|www\S+', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def get_top_words(texts, num=20):
    words = []
    for text in texts:
        cleaned = clean_text(text)
        tokens = [w for w in cleaned.split() if w not in STOP_WORDS and len(w) > 2]
        words.extend(tokens)
    return Counter(words).most_common(num)


top_pos = get_top_words(train_df[train_df['label'] == 1]['text'])
top_neg = get_top_words(train_df[train_df['label'] == 0]['text'])

fig, axes = plt.subplots(1, 2, figsize=(16, 8))
if top_pos:
    words_p, counts_p = zip(*top_pos)
    sns.barplot(x=list(counts_p), y=list(words_p), ax=axes[0], color='green')
    axes[0].set_title('Top 20 Cuvinte - POZITIV', fontsize=14, fontweight='bold')

if top_neg:
    words_n, counts_n = zip(*top_neg)
    sns.barplot(x=list(counts_n), y=list(words_n), ax=axes[1], color='red')
    axes[1].set_title('Top 20 Cuvinte - NEGATIV', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/03_top_cuvinte.png", dpi=300, bbox_inches='tight')
plt.close()


# TOKENIZARE

train_df['text_clean'] = train_df['text'].apply(clean_text)
test_df['text_clean'] = test_df['text'].apply(clean_text)


class SimpleTokenizer:
    def __init__(self, max_vocab=10000, min_freq=2):
        self.max_vocab = max_vocab
        self.min_freq = min_freq
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}
        self.vocab_size = 2

    def build_vocab(self, texts):
        word_counts = Counter()
        for text in texts:
            word_counts.update(text.split())

        filtered = [(w, c) for w, c in word_counts.items() if c >= self.min_freq]
        filtered.sort(key=lambda x: x[1], reverse=True)

        for word, _ in filtered[:self.max_vocab - 2]:
            self.word2idx[word] = self.vocab_size
            self.idx2word[self.vocab_size] = word
            self.vocab_size += 1

        print(f"Vocabular: {self.vocab_size:,} cuvinte")
        return self

    def encode(self, text):
        return [self.word2idx.get(word, 1) for word in text.split()]


tokenizer = SimpleTokenizer(max_vocab=10000, min_freq=2)
tokenizer.build_vocab(train_df['text_clean'])

train_sequences = [tokenizer.encode(text) for text in train_df['text_clean']]
test_sequences = [tokenizer.encode(text) for text in test_df['text_clean']]


# PADDING

seq_lengths = [len(seq) for seq in train_sequences]
MAX_LEN = int(np.percentile(seq_lengths, 95))
print(f"Max sequence length: {MAX_LEN}")


def pad_sequences(sequences, maxlen, value=0):
    padded = np.zeros((len(sequences), maxlen), dtype=np.int32)
    for i, seq in enumerate(sequences):
        if len(seq) > 0:
            trunc = seq[:maxlen]
            padded[i, :len(trunc)] = trunc
    return padded


train_padded = pad_sequences(train_sequences, MAX_LEN)
test_padded = pad_sequences(test_sequences, MAX_LEN)

print(f"Train shape: {train_padded.shape}")
print(f"Test shape: {test_padded.shape}")


# EMBEDDINGS (Word2Vec)

try:
    from gensim.models import Word2Vec

    train_corpus = [text.split() for text in train_df['text_clean']]
    w2v_model = Word2Vec(train_corpus, vector_size=100, window=5, min_count=2, workers=4, sg=0, epochs=20)

    EMBEDDING_DIM = 100
    embedding_matrix = np.zeros((tokenizer.vocab_size, EMBEDDING_DIM))

    for word, idx in tokenizer.word2idx.items():
        if word in w2v_model.wv:
            embedding_matrix[idx] = w2v_model.wv[word]
        else:
            embedding_matrix[idx] = np.random.normal(0, 0.1, EMBEDDING_DIM)

    print(f"Embedding matrix: {embedding_matrix.shape}")

except ImportError:
    print("No gensim")
    EMBEDDING_DIM = 100
    embedding_matrix = np.random.normal(0, 0.1, (tokenizer.vocab_size, EMBEDDING_DIM))
    embedding_matrix[0] = 0

np.save(f'{OUTPUT_DIR}/embedding_matrix.npy', embedding_matrix)
np.save(f'{OUTPUT_DIR}/train_padded.npy', train_padded)
np.save(f'{OUTPUT_DIR}/test_padded.npy', test_padded)
np.save(f'{OUTPUT_DIR}/train_labels.npy', train_df['label'].values)
np.save(f'{OUTPUT_DIR}/test_labels.npy', test_df['label'].values)

with open(f'{OUTPUT_DIR}/tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)


# ANTRENARE MODELE

print(" ANTRENARE MODELE RNN ȘI LSTM")

X_train = train_padded
y_train = train_df['label'].values
X_test = test_padded
y_test = test_df['label'].values

VOCAB_SIZE = tokenizer.vocab_size

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1)


def plot_history(histories, labels, title, filename):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    for history, label in zip(histories, labels):
        axes[0].plot(history.history['accuracy'], label=f'{label} - Train', marker='o')
        axes[0].plot(history.history['val_accuracy'], label=f'{label} - Val', marker='s', linestyle='--')
    axes[0].set_title(f'{title} - Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoca')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    for history, label in zip(histories, labels):
        axes[1].plot(history.history['loss'], label=f'{label} - Train', marker='o')
        axes[1].plot(history.history['val_loss'], label=f'{label} - Val', marker='s', linestyle='--')
    axes[1].set_title(f'{title} - Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoca')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/{filename}', dpi=300, bbox_inches='tight')
    plt.close()



# RNN Baseline

model_rnn1 = Sequential([
    Embedding(VOCAB_SIZE, EMBEDDING_DIM, weights=[embedding_matrix], trainable=False),
    SimpleRNN(64),
    Dense(1, activation='sigmoid')
], name='RNN_Baseline')

model_rnn1.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
history_rnn1 = model_rnn1.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=64,
                              callbacks=[early_stop, reduce_lr], verbose=1)
eval_rnn1 = model_rnn1.evaluate(X_test, y_test, verbose=0)
print(f"Test Acc: {eval_rnn1[1] * 100:.2f}%")


# RNN Regularized

model_rnn2 = Sequential([
    Embedding(VOCAB_SIZE, EMBEDDING_DIM, weights=[embedding_matrix], trainable=True),
    SimpleRNN(128, dropout=0.3, recurrent_dropout=0.2),
    BatchNormalization(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
], name='RNN_Regularized')

model_rnn2.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
history_rnn2 = model_rnn2.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=64,
                              callbacks=[early_stop, reduce_lr], verbose=1)
eval_rnn2 = model_rnn2.evaluate(X_test, y_test, verbose=0)
print(f"Test Acc: {eval_rnn2[1] * 100:.2f}%")


# LSTM Baseline

model_lstm1 = Sequential([
    Embedding(VOCAB_SIZE, EMBEDDING_DIM, weights=[embedding_matrix], trainable=False),
    LSTM(64),
    Dense(1, activation='sigmoid')
], name='LSTM_Baseline')

model_lstm1.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
history_lstm1 = model_lstm1.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=64,
                                callbacks=[early_stop, reduce_lr], verbose=1)
eval_lstm1 = model_lstm1.evaluate(X_test, y_test, verbose=0)
print(f"Test Acc: {eval_lstm1[1] * 100:.2f}%")


# LSTM Regularized

model_lstm2 = Sequential([
    Embedding(VOCAB_SIZE, EMBEDDING_DIM, weights=[embedding_matrix], trainable=True),
    LSTM(128, dropout=0.3, recurrent_dropout=0.2),
    BatchNormalization(),
    Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
], name='LSTM_Regularized')

model_lstm2.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
history_lstm2 = model_lstm2.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=64,
                                callbacks=[early_stop, reduce_lr], verbose=1)
eval_lstm2 = model_lstm2.evaluate(X_test, y_test, verbose=0)
print(f"Test Acc: {eval_lstm2[1] * 100:.2f}%")


# LSTM Bidirectional

model_lstm3 = Sequential([
    Embedding(VOCAB_SIZE, EMBEDDING_DIM, weights=[embedding_matrix], trainable=True),
    Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2)),
    Dense(1, activation='sigmoid')
], name='BiLSTM')

model_lstm3.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
history_lstm3 = model_lstm3.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=64,
                                callbacks=[early_stop, reduce_lr], verbose=1)
eval_lstm3 = model_lstm3.evaluate(X_test, y_test, verbose=0)
print(f"Test Acc: {eval_lstm3[1] * 100:.2f}%")


# Stacked BiLSTM

model_lstm4 = Sequential([
    Embedding(VOCAB_SIZE, EMBEDDING_DIM, weights=[embedding_matrix], trainable=True),
    Bidirectional(LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.2)),
    Bidirectional(LSTM(64, dropout=0.3, recurrent_dropout=0.2)),
    BatchNormalization(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
], name='Stacked_BiLSTM')

model_lstm4.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
history_lstm4 = model_lstm4.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=64,
                                callbacks=[early_stop, reduce_lr], verbose=1)
eval_lstm4 = model_lstm4.evaluate(X_test, y_test, verbose=0)
print(f"Test Acc: {eval_lstm4[1] * 100:.2f}%")


# CNN-LSTM Hybrid

model_lstm5 = Sequential([
    Embedding(VOCAB_SIZE, EMBEDDING_DIM, weights=[embedding_matrix], trainable=True),
    Conv1D(128, 5, activation='relu'),
    MaxPooling1D(2),
    LSTM(64, dropout=0.2, recurrent_dropout=0.2),
    Dense(1, activation='sigmoid')
], name='CNN_LSTM')

model_lstm5.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
history_lstm5 = model_lstm5.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=64,
                                callbacks=[early_stop, reduce_lr], verbose=1)
eval_lstm5 = model_lstm5.evaluate(X_test, y_test, verbose=0)
print(f"Test Acc: {eval_lstm5[1] * 100:.2f}%")


# GRAFICE SI SUMAR

plot_history([history_rnn1, history_rnn2], ['Baseline', 'Regularized'], 'RNN Comparison', '01_rnn_comparison.png')
plot_history([history_lstm1, history_lstm2, history_lstm3, history_lstm4, history_lstm5],
             ['Baseline', 'Regularized', 'BiLSTM', 'Stacked BiLSTM', 'CNN-LSTM'],
             'LSTM Comparison', '02_lstm_comparison.png')

results = pd.DataFrame({
    'Model': ['RNN Baseline', 'RNN Regularized', 'LSTM Baseline', 'LSTM Regularized', 'BiLSTM', 'Stacked BiLSTM',
              'CNN-LSTM'],
    'Test Accuracy': [eval_rnn1[1], eval_rnn2[1], eval_lstm1[1], eval_lstm2[1], eval_lstm3[1], eval_lstm4[1],
                      eval_lstm5[1]],
    'Test Loss': [eval_rnn1[0], eval_rnn2[0], eval_lstm1[0], eval_lstm2[0], eval_lstm3[0], eval_lstm4[0], eval_lstm5[0]]
}).sort_values('Test Accuracy', ascending=False)


print(results.to_string(index=False))
results.to_csv(f'{RESULTS_DIR}/results_summary.csv', index=False)

print(f"Cel mai bun model: {results.iloc[0]['Model']} ({results.iloc[0]['Test Accuracy'] * 100:.2f}%)")



# CONFUSION MATRIX - CEL MAI BUN MODEL


best_idx = results['Test Accuracy'].idxmax()
best_model_name = results.loc[best_idx, 'Model']

models_dict = {
    'RNN Baseline': model_rnn1,
    'RNN Regularized': model_rnn2,
    'LSTM Baseline': model_lstm1,
    'LSTM Regularized': model_lstm2,
    'BiLSTM': model_lstm3,
    'Stacked BiLSTM': model_lstm4,
    'CNN-LSTM': model_lstm5
}

best_model = models_dict[best_model_name]

y_pred_proba = best_model.predict(X_test)
y_pred = (y_pred_proba > 0.5).astype(int).flatten()

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negativ', 'Pozitiv'],
            yticklabels=['Negativ', 'Pozitiv'])
plt.title(f'Confusion Matrix - {best_model_name}', fontsize=14, fontweight='bold')
plt.ylabel('Adevărat')
plt.xlabel('Prezis')
plt.tight_layout()
plt.savefig(f'{RESULTS_DIR}/03_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
