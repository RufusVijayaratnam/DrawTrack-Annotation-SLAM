import numpy as np

class Colour():
    def __init__(self, name):
        self.name = name
        if name == "blue":
            self.lower = np.array([100,100,100],np.uint8)
            self.upper = np.array([140,255,255],np.uint8)
            self.colour = blue = (255, 0, 0)
            self.id = 1.0
        elif name == "yellow":
            self.lower = np.array([20, 100, 100])
            self.upper = np.array([30, 255, 255])
            self.colour = yellow = (0, 255, 255)
            self.id = 2.0
        elif name == "ambiguous":
            self.colour = (0, 0, 255)
            self.id = 0.0
        else:
            self.colour = (0, 0, 0)
            self.id = -1.0


blue = Colour("blue")
yellow = Colour("yellow")
ambiguous = Colour("ambiguous")