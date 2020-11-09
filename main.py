#!/usr/bin/python3

"""
@authors: Michael Albert (albertmichael746@gmail.com)

Creates a GUI in which user input is scraped every X seconds and fed into a
convolutional neural network to determine which digit is being drawn.

For usage, run with the -h flag.

Disclaimers:
- Distributed as-is.
- Please contact me if you find any issues with the code.

"""

import torch
import sys
import time
import skimage.measure
import numpy as np
import nn
import paint

def main(argv):
    model = nn.load(argv[1])
    gui = paint.gui()
    print("past init")
    while True:
        time.sleep(2 - time.monotonic() % 2)
        image = gui.get_image()
        data = np.array(image).astype(np.float32)
        data = skimage.measure.block_reduce(data, block_size=(10,10),func=np.mean)
        data = torch.from_numpy(data)
        data = data[None,None,:,:] # correct the number of dimensions
        print(data.size())
        nn.predict(model, data)

if __name__ == "__main__":
    main(sys.argv)
