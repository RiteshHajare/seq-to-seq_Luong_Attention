import numpy as np
from tqdm import tqdm   

def sigmoid(input, derivative = False):
    if derivative:
        return input * (1 - input)

    return 1 / (1 + np.exp(-input))

def tanh(input, derivative = False):
    if derivative:
        return 1 - input ** 2

    return np.tanh(input)

def softmax(input,derivative=False):
  x = np.asarray(input)
  if x.ndim == 1:
    shifted = x - np.max(x)
    exp = np.exp(shifted)
    sm = exp / np.sum(exp)
  else:
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    sm = exp / np.sum(exp, axis=-1, keepdims=True)
  if derivative:
    return sm * (1 - sm)
  return sm

def initWeights(input_size, output_size):
    input_size,output_size = output_size,input_size
    return np.random.uniform(-1, 1, (output_size, input_size)) * np.sqrt(6 / (input_size + output_size))


class LSTM:
  def __init__(self,vocab_size,hidden_size,learning_rate,output_dims):
    self.input_size = vocab_size + hidden_size
    self.vocab_size = vocab_size
    self.hidden_size = hidden_size
    self.learning_rate = learning_rate
    self.output_dims = output_dims

    # Forget Gate
    self.wf = initWeights(self.input_size,hidden_size)
    self.bf = np.zeros((1,hidden_size))

    # Input Gate
    self.wi = initWeights(self.input_size,hidden_size)
    self.bi = np.zeros((1,hidden_size))

    # Candidate Gate
    self.wc = initWeights(self.input_size,hidden_size)
    self.bc = np.zeros((1,hidden_size))

    # Output Gate
    self.wo = initWeights(self.input_size,hidden_size)
    self.bo = np.zeros((1,hidden_size))

    # Final Gate
    self.wfi = initWeights(2*hidden_size,output_dims)
    self.bfi = np.zeros((1,output_dims))

    # General attention weight matrix
    self.Wa = initWeights(hidden_size, hidden_size)   # [H, H]

  def setParams(self,lstm):
     self.lstm=lstm

  def reset(self,lstm):
    self.forget_output = {}
    self.input = {}
    self.input_output = {}
    self.candidate_output = {}
    self.output_output = {}
    self.long_mem = {-1:lstm.long_mem[max(lstm.long_mem.keys())] }
    self.short_mem = {-1:lstm.short_mem[max(lstm.short_mem.keys())]}
    self.final_output = {}

    self.dforget_output = np.zeros_like(self.wf)
    self.dinput_output = np.zeros_like(self.wi)
    self.dcandidate_output = np.zeros_like(self.wc)
    self.doutput_output = np.zeros_like(self.wo)
    self.dlong_mem = {}
    self.dshort_mem = {}
    self.dfinal_output = np.zeros_like(self.wfi)
    self.dinput = []

    self.dbf = np.zeros_like(self.bf)
    self.dbi = np.zeros_like(self.bi)
    self.dbc = np.zeros_like(self.bc)
    self.dbo = np.zeros_like(self.bo)
    self.dbfi = np.zeros_like(self.bfi)
    self.dWa = np.zeros_like(self.Wa)

    self.attn_weights={}
    self.context ={}
  
  # Add this to your LSTM decoder class
  def compute_attention(self, decoder_hidden, encoder_outputs):
    # encoder_outputs: list of encoder hidden states (shape: [seq_len, hidden_size])
    # decoder_hidden: current decoder hidden state (shape: [1, hidden_size])
    transformed = np.dot(decoder_hidden, self.Wa)  # [seq_len, hidden_size]
    scores = np.dot(encoder_outputs, transformed.T).squeeze()  # [seq_len]
    attn_weights = softmax(scores)  # [seq_len]
    context = np.sum(encoder_outputs * attn_weights[:, None], axis=0, keepdims=True)  # [1, hidden_size]
    return context, attn_weights

  def forward(self,input,idx):

    self.input[idx] = np.concatenate((self.short_mem[idx-1],input),axis=1)

    #Forget Output
    self.forget_output[idx] = sigmoid(np.dot(self.input[idx],self.wf) + self.bf)

    #Input Output
    self.input_output[idx] = sigmoid(np.dot(self.input[idx],self.wi) + self.bi)

    #Candidate Output
    self.candidate_output[idx] = tanh(np.dot(self.input[idx],self.wc) + self.bc)

    #OutputGate Output
    self.output_output[idx] = sigmoid(np.dot(self.input[idx],self.wo) + self.bo)

    self.long_mem[idx] = self.forget_output[idx]*self.long_mem[idx-1] + self.input_output[idx]*self.candidate_output[idx]
    self.short_mem[idx] = tanh(self.long_mem[idx])*self.output_output[idx]

    encoder_outputs = np.vstack(self.lstm.encoder_outputs)
    context, attn_weights = self.compute_attention(self.short_mem[idx], encoder_outputs)
    self.context[idx] = context  # Store context for use in backward
    self.attn_weights[idx] = attn_weights
    combined = np.concatenate((self.short_mem[idx], context), axis=1)  # shape: [1, hidden_size*2]
    # Pass 'combined' to your output layer instead of just decoder_hidden
    self.final_output[idx] = softmax(np.dot(combined, self.wfi) + self.bfi)

    return self.final_output[idx]
    

  def backward(self,idx,targetIdx):
    dlongnext = self.dlong_mem.get(idx + 1, np.zeros((1, self.hidden_size)) )
    dshortnext = self.dshort_mem.get(idx + 1, np.zeros((1, self.hidden_size)) )

    dvalues = np.copy(self.final_output[idx])
    dvalues = -dvalues
    dvalues[0,targetIdx] += 1

    context = self.context[idx]  # Retrieve context from forward
    combined = np.concatenate((self.short_mem[idx], context), axis=1)  # shape: [1, hidden_size*2]
    self.dfinal_output += np.dot(combined.T, dvalues)
    self.dbfi += dvalues
    grad = np.dot(dvalues, self.wfi.T)
    tobackProp = grad[:, :self.hidden_size]
    dcontext = grad[:, self.hidden_size:]

    attn_weights = self.attn_weights[idx]          # [seq_len]
    encoder_outputs = np.vstack(self.lstm.encoder_outputs)    # [seq_len, hidden_size]

    # 1. Gradient w.r.t attn_weights from context
    # context = sum_t(attn_weights[t] * encoder_outputs[t])
    d_attn_weights = np.dot(encoder_outputs, dcontext.T).squeeze()  # [seq_len]

    # 2. Backprop through softmax
    # d_scores = attn_weights * (d_attn_weights - sum(d_attn_weights * attn_weights))
    d_scores = attn_weights * (d_attn_weights - np.sum(d_attn_weights * attn_weights))  # [seq_len]

    # 3. Gradient to encoder_outputs from the score computation
    # scores = encoder_outputs @ decoder_hidden.T
    transformed_hidden = np.dot(self.short_mem[idx], self.Wa)
    for t in range(len(attn_weights)):
        self.lstm.dencoder_outputs[t] += (
            attn_weights[t] * dcontext          # path 1: you already had this
            + d_scores[t] * transformed_hidden  # path 2: score path for Luong general
        )

    # 4. Gradient to decoder hidden state (short_mem) - also missing
    # scores = encoder_outputs @ decoder_hidden.T, so d_hidden = encoder_outputs.T @ d_scores
    dtransformed = np.dot(d_scores[np.newaxis, :], encoder_outputs)  # [1, hidden_size]
    self.dWa += np.dot(self.short_mem[idx].T, dtransformed)  # [H, H]
    d_hidden_from_attn = np.dot(dtransformed, self.Wa.T)  # [1, hidden_size]

    # Then add this into tobackProp before continuing the rest of backward
    tobackProp += d_hidden_from_attn
    
    # for t in range(len(attn_weights)):
    #     self.lstm.dencoder_outputs[t] += attn_weights[t] * dcontext

    dh_temp = tobackProp + dshortnext

    d_o = tanh(self.long_mem[idx]) * dh_temp * sigmoid(self.output_output[idx],derivative=True)
    self.doutput_output += np.dot(self.input[idx].T,d_o)
    self.dbo += d_o

    dc_temp = tanh(tanh(self.long_mem[idx]),derivative=True) * self.output_output[idx] * dh_temp + dlongnext

    d_f = dc_temp * self.long_mem[idx-1] * sigmoid(self.forget_output[idx],derivative=True)
    self.dforget_output += np.dot(self.input[idx].T,d_f)
    self.dbf += d_f

    d_i = dc_temp * self.candidate_output[idx] * sigmoid(self.input_output[idx],derivative=True)
    self.dinput_output += np.dot(self.input[idx].T,d_i)
    self.dbi += d_i

    d_c = dc_temp * self.input_output[idx] * tanh(self.candidate_output[idx],derivative=True)
    self.dcandidate_output += np.dot(self.input[idx].T,d_c)
    self.dbc += d_c

    dz = np.dot(d_f,self.wf.T) + np.dot(d_c,self.wc.T) + np.dot(d_i,self.wi.T) + np.dot(d_o,self.wo.T)

    self.dshort_mem[idx] = np.clip(dz[:, :self.hidden_size], -1, 1)
    self.dlong_mem[idx] = np.clip(self.forget_output[idx] * dc_temp, -1, 1)
    self.dinput.insert(0, np.clip(dz[:, self.hidden_size:],-1,1))

  def optimise(self):
    for param_grad in [self.dforget_output, self.dinput_output, self.dcandidate_output, self.doutput_output, self.dfinal_output,
                       self.dbf, self.dbi, self.dbc, self.dbo, self.dbfi,self.dWa]:
        np.clip(param_grad, -1, 1, out=param_grad)
    self.wf += self.learning_rate*self.dforget_output
    self.wi += self.learning_rate*self.dinput_output
    self.wc += self.learning_rate*self.dcandidate_output
    self.wo += self.learning_rate*self.doutput_output
    self.wfi += self.learning_rate*self.dfinal_output

    self.Wa += self.learning_rate * self.dWa

    self.bf += self.learning_rate*self.dbf
    self.bi += self.learning_rate*self.dbi
    self.bc += self.learning_rate*self.dbc
    self.bo += self.learning_rate*self.dbo
    self.bfi += self.learning_rate*self.dbfi
