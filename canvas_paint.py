from tkinter import *
from tkinter.colorchooser import *
import tkinter

import random
from PIL import Image, ImageGrab
import tensorflow as tf

root = tkinter.Tk()

w = tkinter.Canvas( bg="snow")

canvas_width = 500
canvas_height = 500

# Painting on the canvas using rectangles
def x(col):
    def paint(event):
        print (col)
        python_colour= col
        x1, y1 = ( event.x - 4 ), ( event.y - 4 )
        x2, y2 = ( event.x +4 ), ( event.y + 4 )
        w.create_rectangle( x1, y1, x2, y2, fill = python_colour,outline=col)
    w.bind( "<B1-Motion>", paint )

# Takes the colour from the user using th colour chooser
def getColor():
    col =askcolor()
    x(col[1])

# returns the object name (cup/car)
def getName():
    image_path = './test.jpg' # Input image
    
    # Read in the image_data in binary ormat
    image_data = tf.gfile.FastGFile(image_path, 'rb').read()

    # Loading label file, strips off carriage return (removes the "\n" seperating each line)
    label_lines = [line.rstrip() for line
                   in tf.gfile.GFile("retrained_labels.txt")]

    # Unpersists graph from file (loading the graph trained using the cup and car numpies)
    with tf.gfile.FastGFile("retrained_graph.pb", 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

    with tf.Session() as sess:
        # Feed the image_data as input to the graph and get first prediction
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0') #final_result is the output node

        predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data}) #DecodeJpeg/contents:0 is the input node
                                                                                      #running the session;
                                                                                      #storing the output data in the layer "setmax_tensor"

        # Sort to show labels of first prediction in order of confidence
        K=predictions[0]
        top_k = K.argsort()[-len(K):][::-1]  # list of indexes corresponding to the label lines(predictions)
        object_list = []
        for node_id in top_k:
            human_string = label_lines[node_id]
            object_list.append(human_string)
            score = predictions[0][node_id] # percentage of accuracy
            print('%s (score = %.5f)' % (human_string, score))
        name_person = object_list[0]
		
    L = Label(master, text=str(object_list[0]))
    L.grid()
    return name_person

#creating an image of the drawing in the .jpeg format	
def getter(widget):
    x=root.winfo_rootx()+widget.winfo_x()
    y=root.winfo_rooty()+widget.winfo_y()
    x1=x+widget.winfo_width()
    y1=y+widget.winfo_height()
    ImageGrab.grab().crop((x,y,x1,y1)).save("./test.jpg")
    name_object = getName()
    print(name_object)

master = Tk()
master.title( "Paint Application" )
L=Label(master, text="Object Name")
L.grid()

w = Canvas(master, height=canvas_height,width=canvas_width)
w.grid(row=0,columnspan=13)
w.bind( "<B1-Motion>", x("black") )
w.grid()

B=Button(master, text="save", command=lambda:getter(w))
B.grid()

# keeping the text chooser window until the program doesn't end

while True:
    getColor()

message = Label( master, text = """Press and Drag the mouse to draw.
To change color, select the color of your chaoice and press 'OK'""" )
message.grid(columnspan=8)

root.mainloop()

w.update()
