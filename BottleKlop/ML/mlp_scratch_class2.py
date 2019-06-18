

from __future__ import print_function
import mxnet as mx
import numpy as np
from mxnet import nd, autograd, gluon
mx.random.seed(1)
ctx = mx.gpu() if mx.test_utils.list_gpus() else mx.cpu()

isMnist = False






def relu(X):
    return nd.maximum(X, 0)

# ## Dropout

def dropout(X, drop_probability):
    keep_probability = 1 - drop_probability
    mask = nd.random_uniform(0, 1.0, X.shape, ctx=X.context) < keep_probability
    #############################
    #  Avoid division by 0 when scaling
    #############################
    if keep_probability > 0.0:
        scale = (1/keep_probability)
    else:
        scale = 0.0
    return mask * X * scale

def softmax(y_linear):
    exp = nd.exp(y_linear-nd.max(y_linear))
    partition = nd.nansum(exp, axis=0, exclude=True).reshape((-1,1))
    return exp / partition


# ## The *softmax* cross-entropy loss function

def softmax_cross_entropy(yhat_linear, y):
    return - nd.nansum(y * nd.log_softmax(yhat_linear), axis=0, exclude=True)

def evaluate_accuracy(data_iterator, net, num_inputs, num_outputs):
    total_preds = nd.zeros(num_outputs)
    total_labels = nd.zeros(num_outputs)
    total_overlap = nd.zeros(num_outputs)

    numerator = 0.
    denominator = 0.
    for i, (data, label) in enumerate(data_iterator):
      
      if isMnist: 
        data = data.as_in_context(ctx).reshape((-1,num_inputs))
        label = label.as_in_context(ctx)
        output = net(data)
        predictions = nd.argmax(output, axis=1)
        number_same = nd.sum(predictions == label)
      else:
        data = data.as_in_context(ctx).astype(numpy.float32)
        label = label.as_in_context(ctx).astype(numpy.float32)
        output = net(data)

        soft_output = softmax(output)
        single_prediction_index = nd.argmax(soft_output, axis=1)
        
        single_label_index = nd.argmax(label, axis=1)
        number_same = nd.sum(single_prediction_index == single_label_index)

        #output = net(data)
        #predictions = nd.argmax(output, axis=1)
        #numerator += nd.sum(predictions == label)
        numerator += number_same
        denominator += data.shape[0]

        for l in nd.arange(0, num_outputs):

          if True:
            pp = (single_prediction_index == l)
            preds = nd.sum(pp)
            ll = (single_label_index == l)
            labels = nd.sum(ll)
            ss = pp + ll
            overlap = nd.sum(ss>1)
            x = 1
          else:
            preds = 0
            labels = 0
            overlap = 0

            for j in nd.arange(0,batch_size):
              if single_prediction_index[j] == l: preds +=1
              if single_label_index[j] == l: labels +=1
              if (single_prediction_index[j] == l and single_label_index[j] == l \
                and single_prediction_index[j] == single_label_index[j]):
                overlap +=1

            
          ##aaa = numpy.set((single_prediction_index == i).asnumpy())
          ##bbb = numpy.set((single_prediction_index == single_label_index).asnumpy())
          ##ccc = numpy.intersect1d(aaa,bbb)
          ###prediction_nominator[i] +=
          ##prediction_indices = numpy.nonzero((i == single_prediction_index and single_prediction_index == single_label_index).asnumpy())
          ##prediction_set = numpy.set(prediction_indices)
          ###prediction_denominator[single_prediction_index] += 
          #####label_set = numpy.set(i == single_prediction_index)
          total_preds[l] += preds
          total_labels[l] += labels
          total_overlap[l] += overlap


    for l in nd.arange(0, num_outputs):
      p = (total_overlap[l] / total_labels[l]).asscalar()
      r = (total_overlap[l] / total_preds[l]).asscalar() 
      f1 = 1 * (p * r) / (p + r)
      print('f1: %s, l %s, p %s, l: %s, o: %s .. precision: %s, recall: %s'   \
        , (f1 \
          , l.asscalar(), total_preds[l].asscalar() \
          , total_labels[l].asscalar() \
          , total_overlap[l].asscalar() \
          , p \
          , r \
        )        )
      

    return (numerator / denominator).asscalar()


class mlp_scratch:
  def __init__(self, num_inputs = -1, num_outputs = -1,  modelParamsSaved = None, hidden_size1 = 256, hidden_size2 = 128, ):
    if num_inputs == -1 : 
      num_inputs = 150
      num_outputs = 8

    self.modelParamsSaved = modelParamsSaved

    self.num_inputs = num_inputs
    self.num_outputs = num_outputs




    self.layers = [num_inputs, hidden_size1,hidden_size2, num_outputs]

    self.W1 = nd.random_normal(shape=(num_inputs, hidden_size1), ctx=ctx) *.01
    self.b1 = nd.random_normal(shape=hidden_size1, ctx=ctx) * .01

    self.W2 = nd.random_normal(shape=(hidden_size1,hidden_size2), ctx=ctx) *.01
    self.b2 = nd.random_normal(shape=hidden_size2, ctx=ctx) * .01

    self.W3 = nd.random_normal(shape=(hidden_size2,num_outputs), ctx=ctx) *.01
    self.b3 = nd.random_normal(shape=num_outputs, ctx=ctx) *.01

    self.params = [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]

# allocate space for gradients.

    for param in self.params:
        param.attach_grad()

# ## Define the model

  def net(self, X, drop_prob=0.0):
      #######################
      #  Compute the first hidden layer 
      #######################    
      h1_linear = nd.dot(X, self.W1) + self.b1
      h1 = relu(h1_linear)
      h1 = dropout(h1, drop_prob)
    
      #######################
      #  Compute the second hidden layer
      #######################
      h2_linear = nd.dot(h1, self.W2) + self.b2
      h2 = relu(h2_linear)
      h2 = dropout(h2, drop_prob)
    
      #######################
      #  Compute the output layer.
      #  We will omit the softmax function here 
      #  because it will be applied 
      #  in the softmax_cross_entropy loss
      #######################
      yhat_linear = nd.dot(h2, self.W3) + self.b3
      return yhat_linear


# ## Optimizer

# In[ ]:


  def SGD(self, params, lr):    
      for param in self.params:
          param[:] = param - lr * param.grad


  def train(self, train_data, test_data):
    
    epochs = 50
    moving_loss = 0.
    learning_rate = .001

    for e in range(epochs):
        for i, (data, label) in enumerate(train_data):
        
            if isMnist: 
              data = data.as_in_context(ctx).reshape((-1,num_inputs))
              label = label.as_in_context(ctx)
              label_one_hot = nd.one_hot(label, num_inputs)
            else: #make sure that labels some up to 1
              data = data.as_in_context(ctx).astype(numpy.float32)
              label = label.as_in_context(ctx).astype(numpy.float32)
              # make sure labels sum up to one
              label_one_hot = label / numpy.sum(label,1).reshape(batch_size,1)

            with autograd.record():
                ################################
                #   Drop out 50% of hidden activations on the forward pass
                ################################
                output = self.net(data, drop_prob=.5)
                loss = softmax_cross_entropy(output, label_one_hot)
            loss.backward()
            self.SGD(self.params, learning_rate)
        
            ##########################
            #  Keep a moving average of the losses
            ##########################
            if i == 0:
                moving_loss = nd.mean(loss).asscalar()
            else:
                moving_loss = .99 * moving_loss + .01 * nd.mean(loss).asscalar()
            
        test_accuracy = evaluate_accuracy(test_data, self.net, self.num_inputs, self.num_outputs)
        train_accuracy = 1 #evaluate_accuracy(train_data, net)
        print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" % (e, moving_loss, train_accuracy, test_accuracy)) 
        #keep saving params for every epoch
        self.params_save()


  def params_save(self):
    #ndParams = nd.array(params)
    fpath = self.modelParamsSaved # r"c:\temp\params.bin"
    mx.ndarray.save(fpath, self.params)
    #ndParams2 = mx.ndarray.load(fpath)
    #print(numpy.count_nonzero(mx.ndarray.equal(params[0],ndParams2[0]).asnumpy()))

  def params_load(self):
    #ndParams = nd.array(params)
    fpath = self.modelParamsSaved #r"c:\temp\params.bin"
    self.params = mx.ndarray.load(fpath)
    #ndParams2 = mx.ndarray.load(fpath)
    #print(numpy.count_nonzero(mx.ndarray.equal(params[0],ndParams2[0]).asnumpy())) 

    self.W1 = self.params[0]
    self.b1 = self.params[1]

    self.W2 = self.params[2]
    self.b2 = self.params[3]

    self.W3 = self.params[4]
    self.b3 = self.params[5]



  def do1(self, data):
    return self.net(data)

  def describe(self):
    strLayers = '*'.join(str(l) for l in self.layers)
    return strLayers


