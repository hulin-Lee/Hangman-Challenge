import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import heapq
import torch
import torch.utils.data as Data
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.functional import cross_entropy, softmax, relu

import utils
from transformer import Encoder, PositionEmbedding

class HangmanBERT(nn.Module):
    def __init__(self, max_len, emb_dim, n_vocab, num_head, num_layer, drop_rate, padding_index, mask_index, learning_rate):
        super().__init__()
        self.padding_index = padding_index
        self.mask_index = mask_index
        self.n_vocab = n_vocab
        
        # Embedding and Encoder
        self.embed = PositionEmbedding(max_len, emb_dim, n_vocab)  # Combines position and token embeddings
        self.encoder = Encoder(num_head, emb_dim, drop_rate, num_layer)  # Transformer encoder
        self.encoder_dense = nn.Linear(emb_dim, n_vocab)  # Maps encoded representation to vocab size

        # Decoder (used for generating single-query self-attention)
        self.query_embed = nn.Embedding(1, emb_dim)  # Embedding for single query
        self.attn_dense = nn.Linear(emb_dim, emb_dim)  # Dense layer for attention
        self.o_dense = nn.Linear(emb_dim, n_vocab)  # Output dense layer for predictions

        # Optimizer
        self.opt = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def do_mask(self, x):
        """Creates a padding mask for inputs."""
        mask = (x == self.padding_index)  # Boolean mask for padding indices
        return mask[:, None, None, :]  # [batch_size, 1, 1, max_len]

    def forward(self, x, training=False):
        """Forward pass through embedding and encoder."""
        embed = self.embed(x)  # Embedding layer
        mask = self.do_mask(x)  # Padding mask
        encoded = self.encoder(embed, training=True, mask=mask)  # Transformer encoder
        # logits = self.self_attention(encoded, x)  # [batch_size, n_vocab]
        return self.encoder_dense(encoded)  # Maps encoded output to vocab size

    def self_attention(self, encoded, x):
        """Compute single-query self-attention for masked positions."""
        # Generate single query embedding
        device = next(self.parameters()).device
        q = torch.zeros(encoded.size(0), 1, dtype=torch.long).to(device)  # Query indices
        Q = self.attn_dense(self.query_embed(q))  # Query embedding transformation

        # Keys and values are derived from the encoder outputs
        K = encoded
        V = encoded

        # Compute scaled dot-product attention
        score = torch.matmul(Q, K.transpose(1, 2)) / (encoded.shape[-1] ** 0.5)  # Attention scores

        # Mask out positions that are not masked in the input
        mask_positions = (x == self.mask_index)
        score = score.masked_fill(~mask_positions.unsqueeze(1), float('-inf'))  # Masked positions

        # Compute attention weights and context vector
        attn_weights = torch.softmax(score, dim=-1)
        context = torch.matmul(attn_weights, V)  # Weighted sum of values

        # Generate output logits
        output = self.o_dense(context).squeeze(1)
        return output

    def step(self, x, y, training=False):
        """
        Performs a training step: forward pass, loss computation, and backpropagation.
        x: Input tensor [batch_size, max_len]
        y: Target tensor [batch_size, max_len]
        """
        self.opt.zero_grad()  # Clear gradients

        # Forward pass
        encoder_dense = self.forward(x, training) 

        # Select logits and targets corresponding to masked positions
        masked = torch.eq(x, self.mask_index).unsqueeze(2)
        encoder_masked = torch.masked_select(encoder_dense, masked).reshape(-1, encoder_dense.size(-1))
        y_masked = torch.masked_select(y, masked.squeeze(2))
        letter_loss = cross_entropy(encoder_masked, y_masked)  # Cross-entropy loss

        # calculate the loss based on KL divergence
        # y_dist = self.letter_distribution(x, y)  # [batch_size, n_vocab]
        # probs = softmax(logits, dim=-1)  # [batch_size, n_vocab]
        # probs = probs.clamp(min=1e-8) 
        # y_dist = y_dist.clamp(min=1e-8)
        # word_loss = (y_dist * (torch.log(y_dist) - torch.log(probs))).sum(dim=-1).mean()

        loss = letter_loss
        # Backward and optimizer step
        loss.backward()
        self.opt.step()
        return loss.item()

    def test(self, x, y, training=False):
        """
        Evaluates the model without updating weights.
        x: Input tensor [batch_size, max_len]
        y: Target tensor [batch_size, max_len]
        """
        self.eval()  # Set model to evaluation mode
        with torch.no_grad():
            encoder_dense = self.forward(x, training)

            # Compute loss for masked positions
            masked = torch.eq(x, self.mask_index).unsqueeze(2)
            encoder_masked = torch.masked_select(encoder_dense, masked).reshape(-1, encoder_dense.size(-1))
            y_masked = torch.masked_select(y, masked.squeeze(2))
            letter_loss = cross_entropy(encoder_masked, y_masked)

            # y_dist = self.letter_distribution(x, y)  # [batch_size, n_vocab]
            # probs = softmax(logits, dim=-1)  # [batch_size, n_vocab]
            # probs = probs.clamp(min=1e-8) 
            # y_dist = y_dist.clamp(min=1e-8)
            # word_loss = (y_dist * (torch.log(y_dist) - torch.log(probs))).sum(dim=-1).mean()
  
            loss = letter_loss
        self.train()  # Reset model to training mode
        return loss.item()

    def predict(self, x, training=False):
        """
        Predict probabilities for the masked positions.
        x: Input tensor [1, max_len]
        """
        assert x.shape[0] == 1  # Ensure batch size is 1
        self.eval()
        # with torch.no_grad():
        #     logits, _ = self.forward(x, training)  
        #     probabilities = torch.softmax(logits, dim=-1).squeeze(0)  # [n_vocab]
        with torch.no_grad():
            logits = self.forward(x, training)  
            probabilities = torch.softmax(logits, dim=-1)  # Compute probabilities
            mask_positions = (x == self.mask_index)  # Find masked positions
            masked_probabilities = probabilities[mask_positions]  # Extract probabilities for masked positions
            probabilities = masked_probabilities.mean(dim=0)  # Average probabilities
        self.train()
        return probabilities


def do_mask(seq, length, mask_id, mask_rate):
    """Apply random masking to a single sequence."""
    range = np.arange(length)
    rand_id = np.random.choice(range, size=max(1, int(mask_rate * len(range))), replace=False)
    masked_seq = seq.copy()
    masked_seq[rand_id] = mask_id
    return masked_seq

def random_mask(words, lengths, mask_id, mask_rate):
    """Apply random masking to a batch of sequences."""
    masked_words = [do_mask(words[i], lengths[i], mask_id, mask_rate) for i in range(len(words))]
    return np.array(masked_words)

def obs_transform(obs, max_len, padding_idx):
    """Transform observations into input format with padding."""
    x = [char_to_index[s] for s in obs]
    padding = [padding_idx] * (max_len - len(obs))
    x.extend(padding)
    return np.array(x)

def fraction_policy(length, failed_gass):
    """
    Suggest the next letter to guess based on letter frequencies
    of words of the same length, excluding failed guesses.
    """
    if failed_gass is not None:
        mask = pd.concat([know_words[letter] == 0 for letter in failed_gass] + [know_words['length'] == length], axis=1).all(axis=1)
    else:
        mask = know_words['length'] == length
    match = know_words[mask]
    letter_counts = [match[chr(i)].sum() for i in range(ord('a'), ord('z') + 1)]
    idx = letter_counts.index(np.max(letter_counts))
    return chr(ord('a') + idx)

if __name__ == "__main__":
    PAD_ID = 0
    LEARING_RATE = 1e-4
    DROP_RATE = 0.2
    BATCH_SIZE = 64
    EPOCH = 500
    EMB_DIM = 256
    N_HEAD = 8
    N_LAYER = 4
    TRAIN_RATIO = 0.9
    train_data = utils.know_words_dataset(split='train', train_ratio=TRAIN_RATIO, padding_index=PAD_ID)
    test_data = utils.know_words_dataset(split='test', train_ratio=TRAIN_RATIO, padding_index=PAD_ID)
    N_VOCAB = train_data.n_vocab
    MAX_LEN = train_data.max_len
    MASK_ID = N_VOCAB
    
    model = HangmanBERT(max_len=MAX_LEN, emb_dim=EMB_DIM, n_vocab=N_VOCAB +1, num_head=N_HEAD, num_layer=N_LAYER, 
                    drop_rate=DROP_RATE, padding_index=PAD_ID, mask_index=MASK_ID, learning_rate=LEARING_RATE)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)
    # model.load_state_dict(torch.load('hangman_model.pth'))
    train_loss = []
    test_loss = []
    mask_rate = 0.5
    for epoch in range(EPOCH):
        train_loss_epoch = []
        for batch_index, batch in enumerate(train_loader):
            x, length = batch
            y = x.clone()
            masked_x = random_mask(x.numpy(), length.numpy(), MASK_ID, mask_rate)  
            x = torch.tensor(masked_x, dtype=torch.long, device=device)
            y = y.type(torch.LongTensor).to(device)
            loss = model.step(x, y, training=True)
            train_loss_epoch.append(loss)
        train_loss.append(np.mean(train_loss_epoch))
        
        test_loss_epoch = []
        for batch_index, batch in enumerate(test_loader):
            x, length = batch
            y = x.clone()
            masked_x = random_mask(x.numpy(), length.numpy(), MASK_ID, mask_rate)  
            x = torch.tensor(masked_x, dtype=torch.long, device=device)
            y = y.type(torch.LongTensor).to(device)
            loss = model.test(x, y, training=False)
            test_loss_epoch.append(loss)
        test_loss.append(np.mean(test_loss_epoch))
        
    torch.save(model.state_dict(), 'hangman_model.pth')
    plt.figure(dpi=600)
    plt.plot(train_loss, label='train loss')
    plt.plot(test_loss, label='test loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.grid(alpha=0.3) 
    plt.savefig('loss.png', bbox_inches='tight')
    plt.close()
    
    # validation 
    hangman_challenge = utils.hangman_challenge()
    # know_words = pd.read_csv('known_words.csv')
    char_to_index = {**train_data.v2i, '_': MASK_ID}
    word_length = []
    finished_fraction = []
    successful = []
    for i in range(len(hangman_challenge.dictionary)):
        hangman_challenge.reset(i)
        # I try to gass the first letter of the word according the frenquencies of letters in the words with the length of the target
        # in order to avoid the requirement of the training of full mask cases, but found this is not helpful. 
        # while not hangman_challenge.good_gass:  
        #     gass = fraction_policy(len(hangman_challenge.observation), hangman_challenge.failed_gass)
        #     hangman_challenge.test_policy(gass) 
        while hangman_challenge.left_chance > 0 and (not hangman_challenge.success):   
            obs = obs_transform(hangman_challenge.observation, MAX_LEN, PAD_ID)
            obs = torch.from_numpy(obs).unsqueeze(0).type(torch.LongTensor).to(device)
            probs = model.predict(obs)
            letter_probabilities = probs[1:26].cpu()
            sorted_prob = heapq.nlargest(26, enumerate(letter_probabilities), key=lambda x: x[1])
            indices = [index+1 for index, value in sorted_prob]
            for index in indices:
                if train_data.i2v[index] not in hangman_challenge.successful_gass and train_data.i2v[index] not in hangman_challenge.failed_gass:
                    gass = train_data.i2v[index]
                    break
            hangman_challenge.test_policy(gass)
        word_length.append(len(hangman_challenge.observation))
        finished_fraction.append(hangman_challenge.finished_fraction)
        successful.append(hangman_challenge.success)
    word_length = np.array(word_length)
    finished_fraction = np.array(finished_fraction)
    successful = np.array(successful)
    print('total successful fraction:', np.sum(successful) / len(hangman_challenge.dictionary))
    
    successful_fraction = []
    plt.figure(dpi=600)
    for l in range(4, 16, 1):
        filtered_fraction = finished_fraction[word_length == l]
        successful_fraction.append(np.sum(successful[word_length == l]) / len(successful[word_length == l]))
        mean_value = filtered_fraction.mean()
        lower_error = filtered_fraction.std() 
        upper_error = min(1, filtered_fraction.std() + mean_value)   - mean_value  
        plt.errorbar(l, mean_value, yerr=[[lower_error], [upper_error]], capsize=5, fmt='o')
    filtered_fraction = finished_fraction[word_length > 15]
    successful_fraction.append(np.sum(successful[word_length > 15]) / len(successful[word_length > 15]))
    mean_value = filtered_fraction.mean()
    lower_error = filtered_fraction.std() 
    upper_error = min(1, filtered_fraction.std() + mean_value)   - mean_value  
    plt.errorbar(16, mean_value, yerr=[[lower_error], [upper_error]], capsize=5, fmt='o')
    plt.plot(np.arange(4, 17, 1), successful_fraction, color='gray', label="successful fraction")
    plt.xlabel("words' length")
    plt.ylabel('finished fraction per word')
    plt.legend()
    plt.text(x=10, y=0.4, s='total successful fraction: ' + str(round(np.sum(successful) / len(hangman_challenge.dictionary), 3)), fontsize=10)
    plt.xticks(np.arange(4, 17, 1), list(np.arange(4, 16, 1).astype(str)) + ['>15'])
    plt.grid(alpha=0.3)
    plt.savefig('validation.png', bbox_inches='tight')
    plt.close()
