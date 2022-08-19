"""
Minimal character-level Vanilla RNN model. Written by Andrej Karpathy (@karpathy)
BSD License
"""
import numpy as np

# data I/O
data = open('inputs.txt', 'r') # should be simple plain text file
content=data.read()

listcontent=content.split('\n')
#print listcontent[0]
inputs1=[]
vocab_size=32
hidden_size = 8 # size of hidden layer of neurons
seq_length = 10 # number of steps to unroll the RNN for
learning_rate = 0.05

# model parameters
Wxh = np.random.randn(hidden_size, vocab_size)*0.01 # input to hidden
#print "Wxh "+str(np.shape(Wxh))
Whh = np.random.randn(hidden_size, hidden_size)*0.01 # hidden to hidden forward
#print "whh" +str(np.shape(Whh))
Whhb = np.random.randn(hidden_size, hidden_size)*0.01 #hidden to hidden back
#print "Whhb "+str(np.shape(Whhb))
Why = np.random.randn(vocab_size, hidden_size)*0.01 # hidden to output for
#print "Why "+str(np.shape(Why))
bh = np.zeros((hidden_size, 1)) # hidden bias
by = np.zeros((vocab_size, 1)) # output bias
#exit()

for i in range(1000):
  asd=listcontent[i][1:-1]
  k=asd.split(',')
  k=list(map(int,k))
  inputs1.append(k)

def lossFun(inputs, targets, hprev):
  """
  inputs,targets are both list of integers.
  hprev is Hx1 array of initial hidden state
  returns the loss, gradients on model parameters, and last hidden state
  """
  xs, hs, ys, ps, hsb = {}, {}, {}, {}, {}
  hs[-1] = np.copy(hprev)
  loss = 0
  # forward pass input to hidden
  for t in xrange(len(inputs)):
    xs[t] = np.zeros((vocab_size,1)) # encode in 1-of-k representation
    xs[t][inputs[t]] = 1
    hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh) # hidden state
    
  hsb[len(inputs)] = np.copy(hs[len(inputs)-1])  
 #forward pass forward hidden
  for t in range(len(inputs)-1,-1,-1):
    xs[t] = np.zeros((vocab_size,1)) # encode in 1-of-k representation
    xs[t][inputs[t]] = 1
    hsb[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whhb, hsb[t+1]) + bh) # hidden state

    
#forward pass backward hidden
  hsf=[0.0]*len(inputs)
  ixes=[]
  for t in range(len(inputs)):
    #print hs[t]+hsb[t]
    hsf[t] = hs[t]+hsb[t]
    ys[t] = np.dot(Why, hsf[t]) + by # unnormalized log probabilities for next chars
    #print np.shape(Why)
    #break
    #print ys[t]
    ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t])) # probabilities for next chars
    '''ix = np.random.choice(range(vocab_size), p=ps[t].ravel())
    x1 = np.zeros((vocab_size, 1))
    x1[ix] = 1
    ixes.append(ix)'''
    loss += -np.log(ps[t][targets[t],0]) # softmax (cross-entropy loss)

  #print "Inputs:",inputs," Target",ixes
  #print "-"*35  
 
  
  # backward pass: compute gradients going backwards
  dWxh, dWhh, dWhy, dWhhb = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why),\
  np.zeros_like(Whhb)
  dbh, dby = np.zeros_like(bh), np.zeros_like(by)
  dhnext,dhprev = np.zeros_like(hs[0]),np.zeros_like(hsb[0])
  dy = np.zeros_like(ps[0])

  for t in xrange(len(inputs)):
    dy = np.copy(ps[t])
    dy[targets[t]] -= 1
    #print dy[targets[t]]

    #print targets[t]
   
#back prop

  for t in reversed(xrange(len(inputs)-1)):
    #dy = np.copy(ps[t])
    #dy[targets[t]] -= 1 # backprop into y. see http://cs231n.github.io/neural-networks-case-study/#grad if confused here
    #dWhy += np.dot(dy, hs[t].T)
    #dby += dy
    dh = np.dot(Why.T, dy) + dhnext # backprop into h
    dhraw = (1 - hs[t] * hs[t]) * dh # backprop through tanh nonlinearity
    dbh += dhraw
    dWxh += np.dot(dhraw, xs[t].T)
    dWhh += np.dot(dhraw, hs[t+1].T)
    dhnext = np.dot(Whh.T, dhraw)
    


  for t in xrange(1,len(inputs)):
    #dy = np.copy(ps[t])
    #dy[targets[t]] -= 1 # backprop into y. see http://cs231n.github.io/neural-networks-case-study/#grad if confused here
    #dWhyb += np.dot(dy, hsb[t].T)
    #dby += dy
    dh = np.dot(Why.T, dy) + dhprev # backprop into h
    dhraw = (1 - hsb[t] * hsb[t]) * dh # backprop through tanh nonlinearity
    dbh += dhraw
    dWxh += np.dot(dhraw, xs[t].T)
    dWhhb += np.dot(dhraw, hsb[t-1].T)
    dhprev = np.dot(Whhb.T, dhraw)
  

  for dparam in [dWxh, dWhh, dWhy, dbh, dby, dWhhb]:
    np.clip(dparam, 0, 31, out=dparam) # clip to mitigate exploding gradients
  return loss, dWxh, dWhh, dWhhb, dWhy, dbh, dby, hs[0], hsb[len(inputs)-1]


def sample(h,hsb, seed_ix):
  """ 
  sample a sequence of integers from the model 
  h is memory state, seed_ix is seed letter for first time step
  """
  x = np.zeros((vocab_size, 1))
  x[seed_ix] = 1
  ixes = []


  for t in xrange(len(inputs)):
    h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh) # hidden state
    
 #forward pass forward hidden
  for t in range(len(inputs)-1,-1,-1):
    hsb = np.tanh(np.dot(Wxh, x) + np.dot(Whhb, hsb) + bh) # hidden state
    
#forward pass backward hidden
  hsf=[]
  for t in range(len(inputs)):
    hsf = h+hsb
    y = np.dot(Why, hsf) + by # unnormalized log probabilities for next chars
    ps = np.exp(y) / np.sum(np.exp(y)) # probabilities for next chars
    #print max(ps)
    ix = np.random.choice(range(vocab_size), p=ps.ravel())
    x = np.zeros((vocab_size, 1))
    x[ix] = 1
    ixes.append(ix)

  return ixes


n, p = 0, 0
mWxh, mWhh, mWhhb, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Whhb), np.zeros_like(Why)
mbh, mby = np.zeros_like(bh), np.zeros_like(by) # memory variables for Adagrad
smooth_loss = -np.log(1.0/31)*len(inputs1[0]) # loss at iteration 0
l=0

targets = []
epoch = 0
while True:
  epoch=epoch+1
  for i in range(len(inputs1)):
    for j in range(len(inputs1[i])):
      k = inputs1[i][0:j+1]
      inputs = np.asarray(k)
      if i+1 >= len(inputs1) or i==0: 
        hprev = np.zeros((hidden_size,1))
      targets=np.asarray(sorted(k))
      #print inputs,"   ",targets
      loss, dWxh, dWhh, dWhhb, dWhy, dbh, dby, hprev, hnext = lossFun(inputs, targets, hprev)
      smooth_loss = smooth_loss * 0.999 + loss * 0.001
    
    
  
      for param, dparam, mem in zip([Wxh, Whh, Whhb, Why, bh, by], 
                                [dWxh, dWhh, dWhhb, dWhy, dbh, dby], 
                                [mWxh, mWhh, mWhhb ,mWhy, mbh, mby]):
        mem += dparam * dparam
        param += -learning_rate * dparam / np.sqrt(mem + 1e-8)
    #print smooth_loss    
    if smooth_loss<9:
      break
  print 'iter %d, loss: %f' % (epoch, smooth_loss) # print progress
  if smooth_loss<9.0:
    break
    
      
testing = [[3,2,1,5]]
for i in range(len(testing)):
  for j in range(len(testing[i])):
    k = testing[i][0:j+1]
    inputs = np.asarray(k)
    targets = sorted(k)
    final =  sample(hprev,hnext,testing[i][j])
    print "Targets:",targets
    print "Final:",final




