import numpy as np

def softmax(vector):
    nice_vector = vector - vector.max()
    exp_vector = np.exp(nice_vector)
    exp_denominator = np.sum(exp_vector, axis=1)[:, np.newaxis]
    softmax_ = exp_vector / exp_denominator
    return softmax_

def multiplicative_attention(decoder_hidden_state, encoder_hidden_states, W_mult):
    attention_scores = np.linalg.multi_dot([decoder_hidden_state.T, W_mult, encoder_hidden_states])
    softmax_vector = softmax(attention_scores)
    attention_vector = softmax_vector.dot(encoder_hidden_states.T).T
    return attention_vector

def additive_attention(decoder_hidden_state, encoder_hidden_states, v_add, W_add_enc, W_add_dec):
    second_matrix = np.tanh(np.dot(W_add_enc, encoder_hidden_states) + np.dot(W_add_dec, decoder_hidden_state))
    attention_scores = np.dot(v_add.T, second_matrix)
    softmax_vector = softmax(attention_scores)
    attention_vector = softmax_vector.dot(encoder_hidden_states.T).T
    return attention_vector