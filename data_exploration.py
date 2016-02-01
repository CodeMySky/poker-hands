import numpy as np


def count_poker_hand_types(file_in):
    type_dist = [0] * 10
    for line in file_in:
        print(line)
        data = np.fromstring(line, dtype=int, sep=',')
        print data
        break



if __name__ == '__main__':
    file_in = open('poker-hand-training-true.data', 'r')
    count_poker_hand_types(file_in)