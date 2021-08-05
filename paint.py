import pygame, sys
from pygame.locals import *
from pygame.surfarray import array2d  # surface pixel data
from matplotlib import pyplot as plt
import numpy as np
import import_ipynb
import preprocess
import network

from skimage import measure
from skimage import transform


# notes:
# Ones must be written like this: 1
# since that what the model is trained on. (ie straight lines as ones won't work: |)


def main():
    pygame.init()

    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    screen = pygame.display.set_mode((800, 400), 0, 64)
    screen.fill(WHITE)
    pygame.display.set_caption("Calculator")
    font = pygame.font.Font('freesansbold.ttf', 24)

    mouse_position = (0, 0)
    last_pos = None
    drawing = False

    while True:
        for event in pygame.event.get():

            if event.type == QUIT:
                pygame.quit()
                sys.exit()

            elif event.type == MOUSEMOTION:

                if (drawing):
                    mouse_position = pygame.mouse.get_pos()
                    if last_pos is not None:
                        pygame.draw.line(screen, BLACK, last_pos, mouse_position, 8)

                    last_pos = mouse_position

            # Process image here
            elif event.type == MOUSEBUTTONUP:

                drawing = False

                # clears the equation and answer labels
                clear = pygame.Surface((800, 32))
                screen.blit(clear, (0, 0))

                pixels = np.asarray(array2d(screen))
                pixels = np.divide(pixels, 16777215)
                pixels = pixels[:, 32:]  # reserve top of the screen for printing chars
                chars = preprocess.getChars(pixels)  # returns numpy array

                # convert data from (x,y) to (row,col) format for the model.
                for i in range(chars.shape[0]):
                    chars[i] = chars[i].transpose()

                # convert data shape to (m, row, col, n_c)
                chars = chars.reshape(chars.shape[0], network.dsloader.IMG_ROWS, network.dsloader.IMG_COLS, 1)
                # get the index on the predicted label
                preds = np.argmax(network.model.predict(chars), axis=-1)
                preds_str = str()

                for n in preds:
                    preds_str = preds_str + network.dsloader.CLASS_INDEX[int(n)]

                preds_str = preds_str.replace('star', '*').replace('slash', '/')
                equ_label = font.render('equ = ' + preds_str, 1, (0, 255, 0))

                if (preds_str[-1].isdigit() or preds_str[-1] == ')') and preds_str.count('(') == preds_str.count(')'):

                    try:
                        ans_label = font.render('ans = ' + str(eval(preds_str)), 1, (0, 255, 0))
                        screen.blit(ans_label, (200, 0))

                    except:
                        None

                screen.blit(equ_label, (0, 0))

            elif event.type == MOUSEBUTTONDOWN:
                drawing = True
                last_pos = pygame.mouse.get_pos()

        pygame.display.update()

main()


