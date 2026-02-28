import numpy as np
from helper import prepareData
from layers.embedding import Embedding 
from layers import lstm_encoder
from layers import lstm_decoder
from tqdm import tqdm

input_lang, output_lang, pairs = prepareData('eng', 'fra', False)
print(input_lang.name, output_lang.name)
print(pairs[:20])
emb_dims = 25
emb_enc = Embedding(input_lang.n_words,emb_dims)
encoder = lstm_encoder.LSTM(emb_dims,25,0.01,input_lang.n_words)

emb_dec = Embedding(output_lang.n_words,emb_dims)
decoder = lstm_decoder.LSTM(emb_dims,25,0.01,output_lang.n_words)
decoder.setParams(encoder)
encoder.setParams(decoder)
# print(emb.embedding_matrix)
# print("-------------------")

for id in tqdm(range(5000)):
    for i in range(len(pairs)):
        input_seq = [input_lang.word2index[word] for word in pairs[i][0].split(' ')]
        emb_matrix = emb_enc.forward(input_seq)
        encoder.reset(seq_len=len(input_seq))
        for idx,input in enumerate(emb_matrix):
            encoder.forward([input],idx)
        target_seq = [2] + [output_lang.word2index[word] for word in pairs[i][1].split(' ')]
        emb_matrix = emb_dec.forward(target_seq)
        decoder.reset(encoder) 
        for idx,input in enumerate(emb_matrix):
            decoder.forward([input],idx)
        
        y = [output_lang.word2index[word] for word in pairs[i][1].split(' ')] + [3]
        for idx,output in reversed(list(enumerate(y))):
            decoder.backward(idx,output)

        emb_dec.backward(y,decoder.dinput)

        for idx,output in reversed(list(enumerate(input_seq))):
            encoder.backward(idx)

        emb_enc.backward(input_seq,encoder.dinput)
        encoder.optimise()
        decoder.optimise()
    

def test_model(encoder, decoder, input_lang, output_lang,input_word):
    # Generate a new input sentence
    input_sentence = input_word
    input_indices = [input_lang.word2index[word] for word in input_sentence.split(' ')]
    
    # Pass the input sentence through the encoder
    encoder.reset(seq_len=len(input_indices))
    emb_matrix = emb_enc.forward(input_indices)
    for idx, input in enumerate(emb_matrix):
        encoder.forward([input], idx)
    
    # Pass the encoded input through the decoder
    target_seq = [2]  # Start with SOS token only
    decoder.reset(encoder)
    while True:
        emb_matrix = emb_dec.forward(target_seq)
        pred = decoder.forward([emb_matrix[-1]], len(target_seq)-1)
        newIdx = np.argmax(pred.squeeze())
        if newIdx==3 or len(target_seq)>20:  # EOS token or max length reached
            break
        target_seq.append(newIdx)
    
    # Get the predicted output sentence
    predicted_output = []
    
    # Generate output token by token
    for step in range(len(decoder.final_output)):
        if step < len(decoder.final_output):
            output_probs = decoder.final_output[step]
            predicted_token = np.argmax(output_probs)
            
            if predicted_token == 3:  # EOS token
                break
            
            if predicted_token in output_lang.index2word:
                predicted_word = output_lang.index2word[predicted_token]
                predicted_output.append(predicted_word)
    
    # Print results
    print("Input sentence:", input_sentence)
    print("Predicted output:", ' '.join(predicted_output))
    # print("Predicted tokens:", [np.argmax(o) for o in decoder.final_output[:10]])


# After training, call the test_model function
# test_model(encoder, decoder, input_lang, output_lang,"drop it")
# test_model(encoder, decoder, input_lang, output_lang,"i m ok")
# test_model(encoder, decoder, input_lang, output_lang,"i m fat")
# test_model(encoder, decoder, input_lang, output_lang,"i m fit")
# test_model(encoder, decoder, input_lang, output_lang,"i m hit !")
# test_model(encoder, decoder, input_lang, output_lang,"i m ill")
# test_model(encoder, decoder, input_lang, output_lang,"i m sad")
# test_model(encoder, decoder, input_lang, output_lang,"i m shy")
# test_model(encoder, decoder, input_lang, output_lang,"i m wet")
# test_model(encoder, decoder, input_lang, output_lang,"i m ill")
    
test_sentences = [
"he s too old for me",
"he s very skeptical",
"heed public opinion",
"her bicycle is blue",
"her cheeks were red",
"her novel sold well",
"her parents hate me",
"her parents love me",
"her son is a genius",
"here is your change",
"here s your pudding",
"hey you re not tom",
"his bicycle is blue",
"his dog barks at me"
]

for sent in test_sentences:
    test_model(encoder, decoder, input_lang, output_lang, sent)