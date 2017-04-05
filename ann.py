import numpy as np
import random as r
import math as m

# predictable seed
r.seed("The quick brown fox jumped over the yellow dog")

# global rate
adj_rt = 0.1



# class for a sigmoid neuron
class sig_neur:
    def __init__(self, init_wts, adj_rt, mid):
        self.wts = np.array(init_wts, dtype=float)
        self.adj_rt = adj_rt # adjustment rate for backprop
        self.mid = mid # midpoint sets a lower bound on the sigmoid
        self.denom = (1 - mid) * 2
        self.bound = self.denom - 1
        self.range = 1 - self.denom # property to be read

    # does a dot product of the weights and the inputs
    pulse = lambda self, inps: np.dot(inps, self.wts)

    # sigmoid on range [1 - 2*(1 - mid), 1]
    sig = lambda t: (self.denom / (1 + m.e ** (t * -1))) - self.bound
    
    # yields 1 if step function above threshold
    # yields -1 or 0 otherwise
    def activate(self, inps, bipolar=True):
        self.hold = inps # hold for error correction
        return self.sig(self.pulse(np.array(inps, dtype=float)))

    def correct(self, error):
        pass
        



# yields a fully-connected neural network
# returns a list of layers of neurons equal to the integer at each index
# sets the number of weights on the first layer = n_inps
# random initialization
def network(layers, n_inps, adj_rt):
    network = [0 for _ in layers]
    rand_wts = lambda layer: [2 * r.random() - 1 for n in layer]
    network[0] = [st_neur(rand_wts(n_inps * [0]), adj_rt) \
            for _ in range(layers[0])]
    i = 1
    
    while i < len(layers):
        network[i] = [st_neur(rand_wts(network[i-1]), adj_rt) \
                for _ in range(layers[i])]
        i += 1

    return network



# yields a big-endian array of bits corresponding to the number
def int_to_arr(num):
    arr = []
    while num > 0:
        arr.insert(0, num % 2)
        num = num >> 1
    return arr



# renders the output layer as an integer for display
# taken from this nifty stack overflow answer:
# https://stackoverflow.com/questions/12461361/bits-list-to-integer-in-python
def arr_to_int(layer, midpoint):
    bits = 0
    # traverses the list, prepending each bit
    # [1, 0, 0, 1, 0] -> 0b10010 == 18
    for bit in layer:
        bits = bits << 1 | bit > midpoint

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

