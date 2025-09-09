import math
import os
import time
import random

import pygame
from perlin_numpy import generate_fractal_noise_3d
from pydub import AudioSegment
import sys
import numpy as np
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *


SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
CYAN = (0, 163, 255)
NAVY_BLUE = (0, 0, 128)


pygame.init()
pygame.display.set_caption("J.A.R.V.I.S.")
CLOCK = pygame.time.Clock()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), DOUBLEBUF | OPENGL)

gluPerspective(45, (SCREEN_WIDTH / SCREEN_HEIGHT), 0.1, 50.0)
glTranslatef(0.0, 0.0, -25) # Move the camera back to view the scene
glRotatef(-90.0, 1.0, 0.0, 0.0)



#some stuff no clue what it does
width, height = 322, 322
scale = 322 * 0.9 / 4


def audio_analyze(audio):
    song = AudioSegment.from_wav(audio)
    
    magnitude_lists=[]
    magnitude_avg_list=[]
    x_y_z_form = []
    running_total = 0
    ms_cycle_count = 0
    cycle_count = 0
    for i in range(0, len(song.get_array_of_samples()), 50):
        samples = song[i:50+i].get_array_of_samples()

        if len(samples)>0:
            if i < 10:
                magnitude_avg_list.append((np.average(samples), np.average(samples)))
            else:
                magnitude_avg_list.append((np.average(samples), running_total + np.average(samples)))
            running_total += np.average(samples)
            magnitude_lists.append(np.abs(np.fft.fft(samples)))
            x_y_z_form.append([ms_cycle_count, magnitude_avg_list[cycle_count][0], magnitude_avg_list[cycle_count][1]])
            cycle_count += 1
            ms_cycle_count += 50
    return x_y_z_form


def linear_interpolation(remap, old_min, old_max, new_min, new_max):
    return (remap-old_min)/(old_max-old_min)*(new_max-new_min)

class Dot:
    def __init__(self, x, y, color, radius):
        self.pos = (x, y)
        self.color = color
        self.surface = pygame.Surface((radius*2,radius*2), pygame.SRCALPHA)
        pygame.draw.circle(self.surface, self.color, (radius,radius), radius, radius*2)
    def redraw(self, screen, pos = None):
        if pos:
            self.pos = pos
        screen.blit(self.surface, self.pos)


def get_noise_slice(noise_3d, time_offset, noise_amplitude):
   """
   Returns a 2D slice from a pre-generated 3D noise array.
   """
   depth = noise_3d.shape[2]
   slice_index = int(time_offset * 10) % depth
   noise_2d = noise_3d[:, :, slice_index] * noise_amplitude
   normalized_noise = (noise_2d + 1) / 2
   dots = []
   adjudication_threshold = 0.5
   for y, row in enumerate(normalized_noise):
       for x, value in enumerate(row):
           value = normalized_noise[x,y]
           color_value = int(value*255)
           if (random.random()*22) < abs(value) :
               if value > adjudication_threshold:
                    dots.append({'pos':(x,y),'color':color_value,'radius':2})
   adjudicated_dots = []
   for dot in range(len(dots)):
       if dot % 4 == 0:
           adjudicated_dots.append(dots[dot])
       
   return adjudicated_dots




def transform_xyz(xyz):
    out = []
    for i in range(len(xyz)):
        out.append(linear_interpolation(xyz[0][1], 0, 1, .675, .9))
    return out

def main_loop():
    audio = audio_analyze("file_example_WAV_1MG.wav")
    amplitude = transform_xyz(audio)
    running = True
    last_update_time = pygame.time.get_ticks()
    update_interval = 50
    list_index = 0
    SEED = 12345
    rng = np.random.RandomState(SEED)
    WIDTH, HEIGHT = 320, 320
    res_x, res_y, res_z = 8, 8, 8
    adjusted_width = WIDTH - (WIDTH % res_x)
    adjusted_height = HEIGHT - (HEIGHT % res_y)
    depth = 96*2
    
    print("Generating 3D noise volume (this may take a moment)...")
    start_time = time.time()
    pre_calculated_noise = (generate_fractal_noise_3d(
        (adjusted_height, adjusted_width, depth),
        (res_x, res_y, res_z),
        octaves=3,
        persistence=0.5,
        rng=rng
    ) * 5)
    end_time = time.time()
    print(f"Finished generating 3D noise volume in {end_time - start_time} seconds.")
    while running:

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            # --- TODO: Add custom input handling here ---
        # --- TODO: Add UI drawing logic here ---
        current_time = pygame.time.get_ticks()

        if current_time - last_update_time >= update_interval:
            if list_index < len(audio):
                noise_frame = get_noise_slice(
                    pre_calculated_noise,
                    time_offset=list_index*0.1,
                    noise_amplitude=amplitude[list_index]
                )
                # for dot in noise_frame:
                #     theta = (dot['pos'][0] / 100) * 2 * math.pi
                #     phi = (dot['pos'][1] / 100) * 2 * math.pi
                #     # dots.append(Dot(250 + dot['pos'][0] * 30, 250 + 30 * dot['pos'][1], CYAN, 10))
                #     x,y,z = 5 * math.sin(phi) * math.cos(theta), 5 * math.sin(phi) * math.sin(theta), 5 * math.cos(phi)
                #     # print(f"adjusted x: {20*math.sin(phi)*math.cos(theta)}, y: {20*math.sin(phi)*math.sin(theta)}, z: {20*math.cos(phi)}")
                #     print(f"({5 * math.sin(phi) * math.cos(theta)},{5 * math.sin(phi) * math.sin(theta)}, {5 * math.cos(phi)})")
                #     glPointSize(5)  # Set the size of the points
                #     glBegin(GL_POINTS)  # Start drawing points
                #
                #     glColor3f(1.0, 1.0, 1.0)  # Set color to white
                #     glVertex3f(x, y, z)  # Draw the point
                #     glEnd()  # Stop drawing
                #     # pygame.draw.circle(screen, dot['color'], dot['pos'], dot['radius'])
                # print('breakline')
                #
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
                glPointSize(3)
                glBegin(GL_POINTS)
                
                # Loop through the dots and draw them
                for dot in noise_frame:
                    theta = (dot['pos'][0] / 322) * 2 * math.pi
                    phi = (dot['pos'][1] / 322) * 2 * math.pi
                    
                    x, y, z = (
                        5 * math.sin(phi) * math.cos(theta),
                        5 * math.sin(phi) * math.sin(theta),
                        5 * math.cos(phi)
                    )
                    
                    # The color and position are set per vertex
                    color_value = dot['color'] / 255.0
                    glColor3f(color_value, color_value, color_value)
                    
                    glVertex3f(x, y, z)
                
                glEnd()  # End drawing points
                list_index += 1
            # Check if there is more data to show
            # if analysis_index < len(AUDIO_ANALYSIS):
                # dot_y = 500 + AUDIO_ANALYSIS[analysis_index]
                # new.redraw(screen, (500, dot_y))
                # analysis_index += 1
            last_update_time = current_time
        
        pygame.display.flip()
        CLOCK.tick(60)
    
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    
    main_loop()
