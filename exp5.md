import numpy as np

# 1. Corpus
corpus = [
    ['i', 'like', 'deep', 'learning'],
    ['i', 'like', 'nlp'],
    ['i', 'enjoy', 'flying']
]

# Parameters
window_size = 1
embedding_dim = 10
learning_rate = 0.01
epochs = 1000

# 2. Build vocabulary
vocab = set([word for sentence in corpus for word in sentence])
word2idx = {word: idx for idx, word in enumerate(vocab)}
idx2word = {idx: word for word, idx in word2idx.items()}
vocab_size = len(vocab)

# 3. Generate training data (CBOW)
def generate_training_data(corpus, window_size):
    training_data = []
    for sentence in corpus:
        for idx, word in enumerate(sentence):
            target = word2idx[word]
            context = []
            for n in range(-window_size, window_size + 1):
                if n == 0 or idx + n < 0 or idx + n >= len(sentence):
                    continue
                context.append(word2idx[sentence[idx + n]])
            training_data.append((context, target))
    return training_data

training_data = generate_training_data(corpus, window_size)

# 4. Initialize weights
W1 = np.random.rand(vocab_size, embedding_dim)
W2 = np.random.rand(embedding_dim, vocab_size)

# 5. Softmax function
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# One-hot encoding
def one_hot_vector(word_idx):
    vec = np.zeros(vocab_size)
    vec[word_idx] = 1
    return vec

# 6-7. Forward and Backward propagation
for epoch in range(epochs):
    loss = 0
    for context_idxs, target_idx in training_data:
        # Forward pass
        context_vectors = W1[context_idxs]  # shape: (context_window, embedding_dim)
        h = np.mean(context_vectors, axis=0)  # shape: (embedding_dim,)
        u = np.dot(h, W2)  # shape: (vocab_size,)
        y_pred = softmax(u)

        # Calculate loss (cross-entropy)
        y_true = one_hot_vector(target_idx)
        loss -= np.log(y_pred[target_idx])

        # Backpropagation
        error = y_pred - y_true  # shape: (vocab_size,)
        dW2 = np.outer(h, error)  # shape: (embedding_dim, vocab_size)

        # Error to hidden layer
        dW1 = np.dot(W2, error) / len(context_idxs)  # shape: (embedding_dim,)

        # Update weights
        for context_idx in context_idxs:
            W1[context_idx] -= learning_rate * dW1
        W2 -= learning_rate * dW2

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# 9. Get word vector
def get_word_vector(word):
    return W1[word2idx[word]]

# Example usage:
print("\nWord vector for 'nlp':")
print(get_word_vector('nlp'))
