import numpy as np

class Embedding:
    def __init__(self, vocab_size, embedding_dim, learning_rate=0.01):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.embedding_matrix = np.random.rand(self.vocab_size, self.embedding_dim)
        self.dembedding_matrix = np.zeros_like(self.embedding_matrix)
    
    def forward(self, batch_indices):
        return self.embedding_matrix[batch_indices]

    def backward(self,y, doutput):
        dembedding_matrix = np.zeros_like(self.embedding_matrix)

        for idx, word_idx in enumerate(y):
            arr = doutput[idx].squeeze()
            dembedding_matrix[word_idx] += arr
        self.dembedding_matrix = dembedding_matrix

    def optimise(self):
        np.clip(self.dembedding_matrix, -1, 1, out=self.dembedding_matrix)
        self.embedding_matrix += self.learning_rate * self.dembedding_matrix
