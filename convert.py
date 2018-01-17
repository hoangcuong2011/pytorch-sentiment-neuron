import os
import torch
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import torch.nn as nn
import argparse
import models
import math

parser = argparse.ArgumentParser(description='sample.py')

parser.add_argument('-init', default='The meaning of life is ',
                    help="""Initial text """)
parser.add_argument('-load_model', default='',
                    help="""Model filename to load""")                   
parser.add_argument('-seq_length', type=int, default=50,
                    help="""Maximum sequence length""")
parser.add_argument('-temperature', type=float, default=0.4,
                    help="""Temperature for sampling.""")
parser.add_argument('-neuron', type=int, default=0,
                    help="""Neuron to read.""")
parser.add_argument('-overwrite', type=float, default=0,
                    help="""Value used to overwrite the neuron. 0 means don't overwrite.""")
parser.add_argument('-layer', type=int, default=-1,
                    help="""Layer to read. -1 = last layer""")
# GPU
parser.add_argument('-cuda', action='store_true',
                    help="""Use CUDA""")
                    
opt = parser.parse_args()    


def batchify(data, bsz):    
    tokens = len(data.encode())
    #print(tokens)
    ids = torch.LongTensor(tokens)
    token = 0
    for char in data.encode():
        ids[token] = char
        token += 1
    nbatch = ids.size(0) // bsz
    ids = ids.narrow(0, 0, nbatch * bsz)
    ids = ids.view(bsz, -1).t().contiguous()
    return ids        


batch_size = 1

checkpoint = torch.load(opt.load_model)
embed = checkpoint['embed']
rnn = checkpoint['rnn']



with open("test.2016.pe.phrases") as f:
    content = f.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line
content = [x.strip() for x in content] 

file = open("test.2016.pe.phrases.vectoroutput","w") 

seen = {}

for x in content:
    if x=="":
        continue
    #print(x)
    text = batchify(x, batch_size)
    batch = Variable(text)

    if x in seen:
        #I did some thing before
        #print("I did some thing before")
        outputs = seen[x]
        for elements in output:
            #print(elements)
            file.write(str(elements))
            file.write(" ")
        file.write("\n")
        continue



    states = rnn.state0(batch_size)
    if isinstance(states, tuple):
        hidden, cell = states
    else:
        hidden = states
    last = hidden.size(0)-1
    if opt.layer <= last and opt.layer >= 0:
        last = opt.layer

        
    loss_avg = 0
    loss = 0
    gen = bytearray()
    for t in range(text.size(0)):                           
        emb = embed(batch[t])
        ni = (batch[t]).data[0]
        states, output = rnn(emb, states)
        if isinstance(states, tuple):
            hidden, cell = states
        else:
            hidden = states
        if t == len(range(text.size(0)))-1:
            output = hidden.data[last,0]
            if x not in seen:
                #print(" I added ", x)
                seen[x] = output
            #output = feat
            #print(len(output))
            for elements in output:
                #print(elements)
                file.write(str(elements))
                file.write(" ")
            #print("aaaaaoutput ", feat, last)
            file.write("\n")

file.close()