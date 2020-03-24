from __future__ import unicode_literals, print_function, division
from datetime import datetime
from torch.autograd import Variable
from torch import optim
from io import open
import tkinter.font
import tkinter
import torch
import torch.nn as nn
import torch.nn.functional as F
import unicodedata
import string
import re
import csv
import random


TEACHER_FORCING_RATIO = 0.2 
N_ITERS = 50000 
MAX_LENGTH = 50 
HIDDEN_SIZE = 256
LEARNING_RATE = 0.01 
DROPOUT_RATE = 0.1 

class Dic : 
    
    def __init__(self, name): 
        self.name = name
        self.word_to_index = {} 
        self.index_to_word = {} 
        self.word_to_count = {0: "go", 1: "eos"}
        self.n_words = 2 
        
    def input_Sentence_Normal(text):
        sen = re.sub('[?!@#$%^&*().,]', '', text)
        sen = sen.strip()
        return sen    
        
    def add_sentence(self, sentence):
        for word in sentence.split():
            self.add_word(word)
    
    def add_word(self, word): 
        if word not in self.word_to_index:
            self.word_to_index[word] = self.n_words
            self.word_to_count[word] = 1
            self.index_to_word[self.n_words] = word
            self.n_words = self.n_words + 1
        else:
            self.word_to_count[word] = self.word_to_count[word] + 1

    def readText(): 
        data = pd.read_csv('./data/data.csv')
        data.columns = ["text", "code"]
        input_sent, output_sent = data["text"], data["code"]
        inp, outp = Dic('input'), Dic('output')
        input_, output_, pair = [],[],[]
        
        def sentSeparate_text(sents, put_):
            for sent in sents[:-1]:
                sent = str(sent).strip()
                a = list(sent)
                b = list(filter((" ").__ne__, a))
                c = list(filter(("\n").__ne__, b))
                sentence = ""
                for i in c : 
                    sentence = sentence + " " + i
                sentence = sentence.strip()        
                put_.append(sentence)
            return put_
        
        def sentSeparate_code(sents, put_):
            for sent in sents[:-1]:
                sent = sent.split()
                sent = " ".join(sent)
                put_.append(sent)
            return put_
        
        inputs = sentSeparate_text(input_sent, input_)
        outputs = sentSeparate_code(output_sent, output_)

        for i in range(len(inputs)):
            pair.append([inputs[i], outputs[i]])
        
        return inp, outp, pair
        
class Encoder(nn.Module): 
    def __init__(self, input_size, HIDDEN_SIZE):
        super(Encoder, self).__init__()
        self.HIDDEN_SIZE = HIDDEN_SIZE 
        self.embedding = nn.Embedding(input_size, HIDDEN_SIZE)
        self.gru = nn.GRU(HIDDEN_SIZE, HIDDEN_SIZE)
        
    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden
    
    def initHidden(self): 
        result = Variable(torch.zeros(1,1, self.HIDDEN_SIZE))

        return result
        
class Decoder(nn.Module):
    def __init__(self, HIDDEN_SIZE, output_size, dropout_p, max_length=MAX_LENGTH):
        super(Decoder, self).__init__()
        self.HIDDEN_SIZE = HIDDEN_SIZE 
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        
        self.embedding = nn.Embedding(self.output_size, self.HIDDEN_SIZE)
        self.attn = nn.Linear(self.HIDDEN_SIZE * 2 , self.max_length)
        self.attn_combine = nn.Linear(self.HIDDEN_SIZE*2, self.HIDDEN_SIZE)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.HIDDEN_SIZE, self.HIDDEN_SIZE) 
        self.out = nn.Linear(self.HIDDEN_SIZE, self.output_size)
        
    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights
    
    def initHidden(self):
        result = Variable(torch.zeros(1,1,self.HIDDEN_SIZE)) 
        return result
        
def sentence_to_tensor(lang, sentence):
    indexes = []
    for word in sentence.split():
        try :
            indexes.append(lang.word_to_index[word])
        except :
            pass 
    if (len(indexes) == 0) :
        raise StopIteration
    indexes.append(eos)
    result = Variable(torch.LongTensor(indexes).view(-1,1))
    return result        
    
def pair_to_tensor(pair): 
   
    input_tensor = sentence_to_tensor(input_dic, pair[0])
    output_tensor = sentence_to_tensor(output_dic, pair[1])
    return (input_tensor, output_tensor)
    
def train(input_tensor, output_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH): 
    encoder_hidden = encoder.initHidden()
    encoder_optimizer.zero_grad() 
    decoder_optimizer.zero_grad()
    input_length = input_tensor.size()[0]
    target_length = output_tensor.size()[0]
    encoder_outputs = Variable(torch.zeros(max_length, encoder.HIDDEN_SIZE))
    loss = 0 
    
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0][0]

    decoder_input = Variable(torch.LongTensor([[go]]))    
    decoder_hidden = encoder_hidden
    
    if random.random() < TEACHER_FORCING_RATIO : 
        use_teacher_forcing = True  
    else :
        use_teacher_forcing = False

    if use_teacher_forcing:
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss = loss + criterion(decoder_output, output_tensor[di])
            decoder_input = output_tensor[di] 

    else:
        for di in range(target_length): 
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.data.topk(1) 
            ni = topi[0][0]
            decoder_input = Variable(torch.LongTensor([[ni]]))
            loss = loss + criterion(decoder_output, output_tensor[di])
            if ni == eos:
                break

    loss.backward() 
    encoder_optimizer.step()
    decoder_optimizer.step()
    return loss.item() / target_length   
    
def trainIters(encoder, decoder , n_iters , learning_rate):
    
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [pair_to_tensor(random.choice(pairs)) for i in range(n_iters)]
    criterion = nn.NLLLoss()
    
    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        output_tensor = training_pair[1]
        loss = train(input_tensor, output_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        
def answer_chatBot(encoder, decoder, sentence, max_length=MAX_LENGTH):
    
    sentence = Dic.input_Sentence_Normal(sentence)
    input_tensor = sentence_to_tensor(input_dic, sentence)
    input_length = input_tensor.size()[0]
    encoder_hidden = encoder.initHidden()
    encoder_outputs = Variable(torch.zeros(max_length, encoder.HIDDEN_SIZE))
    
    
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_outputs[ei] + encoder_output[0][0]
    decoder_input = Variable(torch.LongTensor([[go]])) 
    
        
    decoder_hidden = encoder_hidden
    decoded_words = [] 
    decoder_attentions = torch.zeros(max_length, max_length)
    
    for di in range(max_length):
        decoder_output, decoder_hidden, decoder_attention = decoder( decoder_input, decoder_hidden, encoder_outputs)
        decoder_attentions[di] = decoder_attention.data
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        
        if ni == eos:
            break
        else:
            decoded_words.append(output_dic.index_to_word[topi.item()])
            
        decoder_input = Variable(torch.LongTensor([[ni]]))
        
    output_words, attentions = decoded_words, decoder_attentions[:di +1]
    output_sentence = ' '.join(output_words)
    
    return output_sentence    
    
#main
input_dic, output_dic, pairs = Dic.readText()
for pair in pairs:
    input_dic.add_sentence(pair[0])
    output_dic.add_sentence(pair[1])

encoder = Encoder(input_dic.n_words, HIDDEN_SIZE)
decoder = Decoder(HIDDEN_SIZE, output_dic.n_words, DROPOUT_RATE)

go, eos = 0, 1 
 
trainIters(encoder, decoder, N_ITERS,LEARNING_RATE)
torch.save(encoder, "50000_256_0.01_0.1_0.2_encoder_통합.pth")
torch.save(decoder, "50000_256_0.01_0.1_0.2_decoder_통합.pth")

save_model_encoder = torch.load("50000_256_0.01_0.1_0.2_encoder_통합.pth")
save_model_decoder = torch.load("50000_256_0.01_0.1_0.2_decoder_통합.pth")
save_model_encoder.eval()
save_model_decoder.eval()

def test(s):
    
    hangul = re.compile('[^ ㄱ-ㅣ가-힣]+') 

    result = hangul.findall(s) 
    result = list(filter((" ").__ne__, result))
    
    l = []
    
    for i in result :
        l.append(i)

    return l

var1 = ["a", "b", "c", "d"]
var2 = ["가", "나", "다" , "라"]

sentence = input()
varible_list = test(sentence)

for i,varible in enumerate(varible_list):
    sentence = sentence.replace(varible,var1[i])

sentence = list(sentence)
sentence = list(filter((" ").__ne__, sentence))
sentence_char = ""


for i in sentence :
    sentence_char = sentence_char + " " + i
    
answer = answer_chatBot(save_model_encoder, save_model_decoder,sentence_char)

for i,varible in enumerate(varible_list) :
    answer = answer.replace(var2[i], varible)

print(answer)
