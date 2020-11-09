#!/usr/bin/python3

"""
@authors: Michael Albert (albertmichael746@gmail.com)

Creates a rudimentary paint window for user input

Disclaimers:
- Distributed as-is.
- Please contact me if you find any issues with the code.

"""

import tkinter as tk
from tkinter import *
import PIL
from PIL import Image, ImageDraw
import threading


b1 = "up"
xold, yold = None, None

def save(image):
    print("saving")
    filename = 'image.png'
    image.save(filename)

def b1down(event):
    global b1
    b1 = "down"

def b1up(event):
    global b1, xold, yold
    b1 = "up"
    xold = None # reset the line when you let go of the button
    yold = None

def motion(event, draw):
    if b1 == "down":
        global xold, yold
        if xold is not None and yold is not None:
            event.widget.create_line(xold,yold,event.x,event.y,smooth=TRUE,width=10)
            draw.line((xold, yold, event.x, event.y), fill='black', width=10)
                          # here's where you draw it. smooth. neat.
        xold = event.x
        yold = event.y

class gui(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.start()

    def run(self):
        # tkinter code
        self.root = tk.Tk()
        self.cv = Canvas(self.root, width=280, height=280)
        self.image = PIL.Image.new('L', (280, 280), 'white')
        self.draw = ImageDraw.Draw(self.image)
        self.cv.bind("<Motion>", lambda event: motion(event, self.draw) )
        self.cv.bind("<ButtonPress-1>", b1down)
        self.cv.bind("<ButtonRelease-1>", b1up)
        self.cv.pack(expand='YES', fill='both')
        self.btn_save = Button(text="save", command = lambda: save(self.image))
        self.btn_save.pack()

        self.root.mainloop()

    def get_image(self):
        return self.image

def main():
    print("Running directly")
    test = gui()
    print("Doing stuff while the window is open")

if __name__ == "__main__":
    main()
