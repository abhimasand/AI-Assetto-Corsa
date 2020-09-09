import pygame

BLACK = pygame.Color('black')
WHITE = pygame.Color('white')

class TextPrint(object):
    def __init__(self):
        self.reset()
        self.font = pygame.font.Font(None, 20)

    def tprint(self, screen, textString):
        textBitmap = self.font.render(textString, True, BLACK)
        screen.blit(textBitmap, (self.x, self.y))
        self.y += self.line_height

    def reset(self):
        self.x = 10
        self.y = 10
        self.line_height = 15

    def indent(self):
        self.x += 10

    def unindent(self):
        self.x -= 10


pygame.init()

# Set the width and height of the screen (width, height).
screen = pygame.display.set_mode((1, 1))

#pygame.display.set_caption("My Game")

# Loop until the user clicks the close button.
done = False

# Used to manage how fast the screen updates.
clock = pygame.time.Clock()

# Initialize the joysticks.
pygame.joystick.init()



i=0

joystick = pygame.joystick.Joystick(i)
joystick.init()

name = joystick.get_name()
axes = joystick.get_numaxes()
buttons = joystick.get_numbuttons()
hats = joystick.get_numhats()

import time


while not done:
    start = time.time()
    for event in pygame.event.get(): # User did something.
        if event.type == pygame.QUIT: # If user clicked close.
            done = True # Flag that we are done so we exit this loop.
        elif event.type == pygame.JOYBUTTONDOWN:
            print("Joystick button pressed.")
        elif event.type == pygame.JOYBUTTONUP:
            print("Joystick button released.")


    #print("Joystick {}".format(i))

    # Get the name from the OS for the controller/joystick.
    
    #print("Joystick name: {}".format(name))

    inputs = []
    # Usually axis run in pairs, up/down for one, and left/right for
    # the other.
    
    #print("Number of axes: {}".format(axes))

    for i in range(axes):
        axis = joystick.get_axis(i)
        inputs.append(axis)
        #print("Axis {} value: {:>6.3f}".format(i, axis))

    
    #print("Number of buttons: {}".format(buttons))

    for i in range(buttons):
        button = joystick.get_button(i)
        inputs.append(button)
        #print("Button {:>2} value: {}".format(i, button))

    
    #print("Number of hats: {}".format(hats))

    for i in range(hats):
        hat = joystick.get_hat(i)
        inputs.append(hat)
        #print("Hat {} value: {}".format(i, str(hat)))

    print (inputs)


    # Limit to 20 frames per second.
    clock.tick(60)
    #print (time.time() - start)



