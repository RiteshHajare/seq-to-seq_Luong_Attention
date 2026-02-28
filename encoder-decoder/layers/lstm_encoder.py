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
  if derivative:
    return softmax(input) * (1 - softmax(input))
  return np.exp(input) / np.sum(np.exp(input))

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
    self.encoder_outputs = []

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
    self.wfi = initWeights(hidden_size,output_dims)
    self.bfi = np.zeros((1,output_dims))

  def setParams(self,lstm):
     self.lstm=lstm

  def reset(self,seq_len):
    self.forget_output = {}
    self.input = {}
    self.input_output = {}
    self.candidate_output = {}
    self.output_output = {}
    self.long_mem = {-1:np.zeros((1,self.hidden_size))}
    self.short_mem = {-1:np.zeros((1,self.hidden_size))}
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

    self.encoder_outputs = []
    self.dencoder_outputs = [np.zeros((1, self.hidden_size)) for _ in range(seq_len)]

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

    self.encoder_outputs.append(self.short_mem[idx].squeeze())
    self.final_output[idx] = softmax(np.dot(self.short_mem[idx],self.wfi) + self.bfi)
    

  def backward(self,idx):
    dlongnext = self.dlong_mem.get(idx + 1, self.lstm.dlong_mem.get(0))
    dshortnext = self.dshort_mem.get(idx + 1,self.lstm.dshort_mem.get(0))

    dh_temp = dshortnext + self.dencoder_outputs[idx]

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
                       self.dbf, self.dbi, self.dbc, self.dbo, self.dbfi]:
        np.clip(param_grad, -1, 1, out=param_grad)
    self.wf += self.learning_rate*self.dforget_output
    self.wi += self.learning_rate*self.dinput_output
    self.wc += self.learning_rate*self.dcandidate_output
    self.wo += self.learning_rate*self.doutput_output
    self.wfi += self.learning_rate*self.dfinal_output

    self.bf += self.learning_rate*self.dbf
    self.bi += self.learning_rate*self.dbi
    self.bc += self.learning_rate*self.dbc
    self.bo += self.learning_rate*self.dbo
    self.bfi += self.learning_rate*self.dbfi
