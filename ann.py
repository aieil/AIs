import numpy as np
import random as r
import math as m
from sys import argv

# predictable seed
r.seed("The quick brown fox jumped over the yellow dog")

# global rate
adj_rt = 0.5

# global midpoint (0.5 -> binary, 0 -> bipolar)
mid = 0.5

# global bits, output will be interpreted as bits
bits = 4

# global usage!!
usage = """
usage: ann <option> <source> (wts_source) [output]

    [output] is a local file name, it will be created if it does not exist
    or overwritten if it does. If no file is specified, results will be
    written to stdout

    (source_2) should only be specified when it is needed

    --help  show help

    --train read a series of light patterns from <source> and train the neural
            network, write trained weights to [output]

    --test  read a series of light patterns from source and a weight set from
            (wts_source) and build a neural network using those weights, test the
            weights on the input patterns and write predictions to [output]
"""


# class for a sigmoid neuron
class sig_neur:
    def __init__(self, init_wts, adj_rt, mid):
        self.wts = np.array(init_wts, dtype=float)
        self.adj_rt = adj_rt # adjustment rate for backprop
        self.mid = mid # midpoint sets a lower bound on the sigmoid
        self.denom = (1 - mid) * 2
        self.bound = self.denom - 1
        self.range = 1 - self.denom # property to be read

    # dot product on inputs, weights
    pulse = lambda self, inps: np.dot(inps, self.wts)

    # sigmoid on range [1 - 2*(1 - mid), 1]
    sig = lambda self, t: (self.denom / (1 + m.e ** (t * -1))) - self.bound
    
    # yields value of sigmoid function for given inputs, stores input
    # values
    def activate(self, inps):
        # hold for error correction
        self.last_inps = inps
        self.last_dot = self.pulse(np.array(inps, dtype=float))
        self.last_out = self.sig(self.last_dot)
        return self.last_out
    
    # corrects using error value, yields error * weight for backprop
    def adjust(self, err):
        # d = dErr/dout * dout/dnet
        d = err * self.last_out * (1 - self.last_out)
        gradient = self.adj_rt * d
        
        # weight adjustment
        i = 0
        while i < len(self.wts):
            self.wts[i] -= (gradient * self.last_inps[i])
            i += 1
        
        # delta * weight[i] set for backprop
        return np.array([d * w for w in self.wts])


# yields a fully-connected neural network
# returns a list of layers of neurons equal to the integer at each index
# sets the number of weights on the first layer = n_inps
# random initialization
def network(layers, n_inps, adj_rt):
    global mid
    network = [0 for _ in layers]
    rand_wts = lambda layer: [2 * r.random() - 1 for n in layer]
    network[0] = [sig_neur(rand_wts(n_inps * [0]), adj_rt, mid) \
            for _ in range(layers[0])]
    i = 1
    
    while i < len(layers):
        network[i] = [sig_neur(rand_wts(network[i-1]), adj_rt, mid) \
                for _ in range(layers[i])]
        i += 1

    return network



# calculates error for each neuron, calls correction method (delta)
def backpropagate(outputs, targets, network):
    output_layer = len(network) - 1
    i = len(network) - 1
    
    # store errors for backpropagation
    errors = []
    # output layer
    j = 0
    while j < len(network[i]):
        # generate initial lists of errors
        errors.append(network[i][j].adjust(outputs[j] - targets[j]))
        j += 1

    # backpropagate
    while i >= 0:
        new_errors = []
        j = 0
        while j < len(network[i]):
            # sum all errors attributable to neuron i,j
            sum_errors = sum(e[j] for e in errors)
            # generate new lists of errors
            new_errors.append(network[i][j].adjust(sum_errors))
            j += 1
        errors = new_errors        
        i -= 1


# just runs the network
def feed_forward(network, inputs):
    for layer in network:
        outputs = [neuron.activate(inputs) for neuron in layer]
        inputs = outputs
    return outputs


# yields a big-endian array of bits corresponding to the number
def int_to_arr(num):
    global bits
    arr = []
    i = 0
    while i < bits:
        arr.insert(0, num % 2)
        num = num >> 1
        i += 1
    return arr



# renders the output layer as an integer for display
# taken from this nifty stack overflow answer:
# https://stackoverflow.com/questions/12461361/bits-list-to-integer-in-python
def arr_to_int(layer):
    global mid
    bits = 0
    # traverses the list, prepending each bit
    # [1, 0, 0, 1, 0] -> 0b10010 == 18
    for bit in layer:
        bits = bits << 1 | (bit > mid)

    return bits



# case of a single output neuron
# subtracts the lower bound to put val on the range [0, denom]
# divides by denom to get range [0, 1]
# then multiplies by n_vals and takes an integer value
# if the maximum value is 9, n_vals = 10 and will yield an integer on (0, 10]
def detector(val, trg_max, n_vals, val_denom):
    return int((val - val_rng) / val_denom * n_vals)



# identifies a light pattern 0-9 using rules
# used in error correction
def identify(light):
    if light == [1, 1, 1, 1, 1, 1, 0]: return 0
    elif light == [0, 1, 1, 0, 0, 0, 0]: return 1
    elif light == [1, 1, 0, 1, 1, 0, 1]: return 2
    elif light == [1, 1, 1, 1, 0, 0, 1]: return 3
    elif light == [0, 1, 1, 0, 0, 1, 1]: return 4
    elif light == [1, 0, 1, 1, 0, 1, 1]: return 5
    elif light == [1, 0, 1, 1, 1, 1, 1]: return 6
    elif light == [1, 1, 1, 0, 0, 0, 0]: return 7
    elif light == [1, 1, 1, 1, 1, 1, 1]: return 8
    elif light == [1, 1, 1, 1, 0, 1, 1]: return 9
    else: return None



# runs backpropagation on all the lines in the test set
def train(network, targets_list):
    global mid
    for t in targets_list:
        outputs = int_to_arr(arr_to_int(feed_forward(net, t)))
        targets = int_to_arr(identify(t))
        backpropagate(outputs, targets, network)



# outputs a feed forward of all the lines in the test set
def test(network, targets_list):
    global mid
    outputs_list = []
    for t in targets_list:
        outputs_list.append(arr_to_int(feed_forward(network, t), mid))
    return outputs_list



# assuming a fit is possible, keeps training until we have a fit
def fit(training_set, network):
    f = open(training_set, 'r')
    light_data = f.readlines()
    f.close()
    light_data = [[int(e) for e in line if e in ('1', '0')] for line in light_data]
    targets_list = [identify(l) for l in light_data]

    outputs_list = test(network, targets_list)
    i = 0
    while outputs_list != targets_list:
        i += 1
        train(network, targets_list)
        outputs_list = test(network, targets_list)

    print("fit target in {} iterations".format(i))



def main():
    n_features = 8 # eight possible corner configurations
    # a common rule of thumb is features^2 neurons on the hidden layer
    # this is simple enough that I don't think I need that many so I'm doing half
    n_hidden = n_features ** 2 / 2
    n_lights = 7 # number of light thingies in display
    global mid
    global adj_rt
    global bits

    if len(argv) > 1:
        if (argv[1] == "--train"):
            net = network([n_features, n_hidden, bits], n_lights, adj_rt, mid)
            fit(argv[2], net)
        else:
            print(usage)
            return
    else:
        print(usage)
        return


if __name__ == "__main__":
    main()
