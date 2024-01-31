# Data.Ml.100 Introduction to Pattern Recognition and Machine Learning
# Md. Mahade Hasan
# Student Number: 50508332
# Exercise 2

#importing essential libraries
from pynput import mouse
import matplotlib.pyplot as plt
import numpy as np

#declating lists for x and y coordinates
x = []
y = []

#mouse click event recogniser
def on_click(a,b,button, pressed):
    if pressed:
        if str(button) == "Button.left":
            x.append(a)
            y.append(b)
        elif str(button) == "Button.right":
            return False

#mouse click listener       
with mouse.Listener(on_click=on_click) as listener:
    print("Please use left button of your mouse to add new points and right button to end")
    listener.join()


# Linear Solver
def my_linfit(x,y):
    
    x_bar = np.mean(x)
    y_bar = np.mean(y)
    
    numerator = 0
    denominator = 0
    
    for i in range(len(x)):
        numerator = numerator + ((y[i]-y_bar)*(x[i]-x_bar))
        denominator = denominator + (x[i]-x_bar)**2
    
    #Implementing the equation of a and b
    a = numerator / denominator
    b = y_bar -(a*x_bar)
    
    return a,b
    
#checking both x and y has the same length
if len(x) !=0 and len(y)!=0 and len(x) == len(y):
    print("The selected points are:")
    for j in range(len(x)):
        print("(",x[j],",",y[j],")")
        
    # checking for atleast 2 points to draw a line. 
    if len(x) >=2 and len(y)>=2:
        a,b = my_linfit(x, y)
        plt.plot(x,y,'go')
        xp = np.arange(0,1000,0.1)
        plt.plot(xp,a*xp+b,'r-')
        plt.axis([0,1000,0,1000])
        print (f"My_fit : a={a}_and_b={b}")
        plt.show()
    else:
        print("You need atleast 2 points to draw a line. ")
