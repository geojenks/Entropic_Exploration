# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 12:59:55 2021

@author: georg

%% uses an algorithmic, state based method to locate maxima within a 3D
matrix. To simulate the tactile exploration of an object

for an example, run find_highs_2D(nearest_empty_2D, stimulus, 1)
"""

# method is a function that operates on an actor (sensor) that is interacting
# with a stimulus (3D matrix) until a finishing condition (state of information) is
# reached

# a policy mpas all history to a new choice

# To do:

# Make predictor, i.e. predict the cell from current knowledge
# Govern control policy from certainty, rather than knowledge (mostly done)

#certainty should be a combination of integrating over the pdf and using neighbouring cells to predict
#the strength of how well cells can predict their neighbours can be trained over multiple data

# Make reward and "punishment" function
# This could be displayed in a matrix too - hot areas would be areas of interest (boundaries)

# is there a critical point for the sd at which one approach becomes more efficient?

# Information (in bits) I = log_2 (1/p) = -log_2(p)  ----p is probability
# we can add bits of sequential information to get overall value for info in bits

#E(I) = sum p(x) * -log_2 (p)
#p is expectancy/ prob of measurement x
#E(I) is expected information from a measurement (entropy), where x is
#distribution of all possible outcomes of the measurement
#Consider greedy (next step) vs global

#Wordle https://www.youtube.com/watch?v=v68zYyaEmEA

import numpy as np
import scipy.integrate as integrate
import pygame
from time import sleep as sleep
import imageio

#import png as rgb array
stimulus = imageio.imread('2D_stim_no_lines_40.png')
# generate random stimulus:
#stimulus = np.random.randint(5,45,size=(10,10))
#knowledge = np.zeros([50,50])
#state = Sensor_state
#state.x = 25
#state.y = 25

# example:     

def find_highs(method, stimulus, render):
    
    # render the original input
    
    # render the exploration/return values
    
    # apply method to stimulus
    
    return locs, time

# trial with 2D, i.e. Minesweeper
res_m = 4 # resolution multiplier

def find_highs_2D(method, stimulus, render):
    sol_space = np.shape(stimulus)
    #print(sol_space)
    res_m = 4 # resolution multiplier
    # obtained values, third index is the 4 RGBA colour, so doesdn't multiply
    knowledge = np.ones((sol_space[0]*res_m, sol_space[1]*res_m, sol_space[2]))*-1
    #  how certain we can be that the knowledge represents the stimulus.
    #  e.g. if 8 values are taken that tightly agree with each other, we don't
    # need the other 8
    # certainty should be inversely proportional to the range of the 95% CI
    certainty = {}
    certainty = np.zeros((sol_space[0], sol_space[1], sol_space[2]))
    #print(certainty)
    '''
    for i in range(0,sol_space[0]):
        for j in range(0, sol_space[1]):
            certainty[i,j] = [0, 0, 0]
    '''
    #certainty = [[0,0,0]]*(sol_space)
    state = Sensor_state
    state.x = round(sol_space[0]*res_m/2)
    state.y = round(sol_space[1]*res_m/2)
    #override
    #state.x = 28
    cell_size = 8
    
    ##### details to visualise the interaction ###########
    # render the original input
    
    background_colour = (255,255,255)
    (width, height) = (((sol_space[0]+3)*5)*cell_size, sol_space[1]*cell_size)
    screen = pygame.display.set_mode((width, height))
    screen.fill(background_colour)
    pygame.display.set_caption('Rendering of knowledge')
    #dictionary
    cell={}
    
    for i in range(sol_space[0],sol_space[0]+sol_space[0]):
        for j in range(sol_space[1]):
            cell[i,j] = Particle(((i+3)*cell_size, j*cell_size), cell_size, stimulus[i-sol_space[0],j])#[0, 0, 0])
            #cell[i,j].update(5*stimulus[i-sol_space[0],j])
            cell[i,j].display(screen)
    pygame.display.flip()
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            
    #######################################################
    
    # apply method to stimulus
    while(1):
        (state, knowledge, certainty) = method(state, knowledge, stimulus, certainty, res_m)
        #### Render the current knowledge state
        for i in range(0,res_m):
            for j in range(0,res_m):
                # make sure we're not looking outside the region
                if state.x + i >= 0 and state.x + i < res_m*sol_space[0] and state.y + j >= 0 and state.y + j < res_m*sol_space[1]:
                    if knowledge[state.x+i,state.y+j][0] != -1:
                        cell[state.x+i,state.y+j] = Particle(((state.x+i)*cell_size/res_m, (state.y+j)*cell_size/res_m), cell_size/res_m, knowledge[state.x+i,state.y+j])
                        #cell[state.x,state.y].update(5*knowledge[state.x,state.y])
                        cell[state.x+i,state.y+j].display(screen)
                        #pygame.display.flip()
                        #sleep(0.01)
        
        #### Render the certainty state, black is known, green is implied,
        # white is unknown
        #space = np.shape(stimulus)
        for i in [2*sol_space[0]+int(state.x/res_m) - 1,
                  2*sol_space[0]+int(state.x/res_m),
                  2*sol_space[0]+int(state.x/res_m) + 1]:
            for j in [int(state.y/res_m)-1,
                      int(state.y/res_m),
                      int(state.y/res_m)+1]:
                # don't draw outside the area
                if i >= 3*sol_space[0] or j >= sol_space[1] or i < 2*sol_space[0] or j < 0:
                    pass
                    #print("passing")
                #else:
                else:
                    #print("not passing")
                    #print(cell_size*int(2*(sol_space[0]+3)+(i+state.x/res_m)), cell_size*int(j/res_m))
                    cell[i,j] = Particle(((i+6)*cell_size,j*cell_size),#(2*(res_m*(sol_space[0]+3)) + cell_size*i,
                                        # cell_size*int(j)),
                                        cell_size,
                                        certainty[i-2*sol_space[0],j])
                    #print(certainty[i-2*sol_space[0],j])
                    cell[i,j].display(screen)
        
        #### Render the averaged colour for each area
        # - this redraws the whole space each iteration, could be more efficient
        #for i in range(sol_space[0]):
        #    for j in range(sol_space[1]):
        for i in [int(state.x/res_m), int(state.x/res_m)]:
            for j in [int(state.y/res_m), int(state.y/res_m)]:
                ### get average over the values
                total = 0
                count = 0
                colour = 0
                #merge the data that lands in the same cell
                for ind1 in range(i*res_m, (i+1)*res_m):
                    for ind2 in range(j*res_m, (j+1)*res_m):
                        if knowledge[ind1,ind2][0] != -1:
                            total = total + knowledge[ind1,ind2]
                            count = count + 1
                if count !=0:
                    colour = total/count
                #print(total)
                #print(colour)
                cell[i,j] = Particle(((i+3*(sol_space[0]+3))*cell_size, j*cell_size), cell_size, colour[0:3])
                #cell[i,j].update(5*colour)
                cell[i,j].display(screen)
        
        #### Render the error between averaged and truth
        # - this redraws the whole space each iteration, could be more efficient
        #for i in range(sol_space[0]):
        #    for j in range(sol_space[1]):
        for i in [int(state.x/res_m), int(state.x/res_m)]:
            for j in [int(state.y/res_m), int(state.y/res_m)]:
                ### get average over the values
                total = 0
                count = 0
                colour = 0
                for ind1 in range(i*res_m, (i+1)*res_m):
                    for ind2 in range(j*res_m, (j+1)*res_m):
                        if knowledge[ind1,ind2][0] !=-1:
                            total = total + knowledge[ind1,ind2]
                            count = count + 1
                if count !=0:
                    colour = abs(stimulus[i,j] - total/count)#*5
                cell[i,j] = Particle(((i+4*(sol_space[0]+3))*cell_size, j*cell_size), cell_size, colour[0:3])
                #print(colour[0:3])
                #cell[i,j].update(5*colour)
                cell[i,j].display(screen)
        
        
        pygame.display.flip()
        # pause so that it's visible
        #sleep(0.05)
        
        Finishing_condition = np.count_nonzero(knowledge==-1) < 3
        if Finishing_condition:
            break
    
    return (state, knowledge)

class Sensor_state:
    x = 0
    y = 0
    z = 0
    rot_x = 0
    rot_y = 0
    rot_z = 0
    reading = 0

######### 2D methods ####################################



# to draw the changing knowledge and stimulus
# This class represents a single cell to render on screen
class Particle:
    def __init__(self, position, size, colour): # ,colour
        self.x, self.y = position
        self.size = size
        self.colour = (255-colour[0], 255-colour[1], 255-colour[2]) # , make this a function of colour
        self.thickness = 0

    def display(self, screen):
        pygame.draw.rect(screen, self.colour, ((self.x, self.y), (self.size, self.size))) #, self.size, self.thickness

    def update(self, R):
        #print(R)
        #self.colour = (5*R, 0, 255 - 5*R)
        self.colour = (R, 0, 255 - R)


#############################

#def method2D(state, knowledge, stimulus):
#    
#    knowledge[state.x, state.y, state.z] = stimulus[state.x, state.y, state.z]
#    
#    return (knowledge, state)
#    if Finishing_condition:
#        return(state)
#    else:
#        method2D(state, knowledge, stimulus)    


def nearest_empty_2D(state, knowledge, stimulus, certainty, res_m):
    (state.x, state.y) = find_nearest_empty(state, knowledge, 0)
    sd = 50
    measurement=[0,0,0,255]
    for i in [0, 1, 2]:
        measurement[i] = np.random.normal(loc=stimulus[int(state.x/res_m), int(state.y/res_m)][i], scale = sd)
        if measurement[i] < 0:
            measurement[i] = 0
        if measurement[i] > 255:
            measurement[i] = 255
    knowledge[state.x, state.y] = measurement
    #knowledge[state.x, state.y] = stimulus[int(state.x/res_m), int(state.y/res_m)]
    for x in [int(state.x/res_m)-1, int(state.x/res_m), int(state.x/res_m)+1]:
        for y in [int(state.y/res_m)-1, int(state.y/res_m), int(state.y/res_m)+1]:
            space = np.shape(stimulus)
            # don't draw outside the area
            if x >= space[0] or y >= space[1] or x < 0 or y < 0:
                pass
            # update the current location equally
            elif x == int(state.x/res_m) and y == int(state.y/res_m):
                #print(certainty[x,y])
                #print(x, y)
                certainty[x, y] = [certainty[x, y][0]+255/(2*res_m*res_m),
                                   certainty[x, y][1]+255/(2*res_m*res_m),
                                   certainty[x, y][2]+255/(2*res_m*res_m),
                                   1]
            # update the neighbours with a green tint
            else:
                #print(certainty[x, y])
                #print(certainty[x, y][0])
                #print(certainty[x, y][1]+255/(32*res_m*res_m))
                #print(certainty[x, y][2])
                certainty[x, y] = [certainty[x, y][0],
                                   certainty[x, y][1]+255/(32*res_m*res_m),
                                   certainty[x, y][2],
                                   1]
    # also colour its neighbours (certainty)
    return (state, knowledge, certainty)

#subsidary function
def find_nearest_empty(state, knowledge, threshold, certforknow=0):
    size = np.shape(knowledge)
    loc = [state.x, state.y]
    #search in rings, dist is 1 off because of index 0
    dist_to_edge = max(state.x, state.y, size[0]-1-state.x, size[1]-1-state.y)
    
    # quick algorithm to find the coords of nearest nodes in ascending order
    radii = []
    for i in range(0, dist_to_edge+1): # to edge
        for j in range(0, i+1):
            radii.append(np.sqrt(pow(i,2)+pow(j,2)))
    radii.sort()
    nodes = []
    # check for unexplored places in each radius
    for rad in radii:
        # list integers up to this distance: x cannot be more than the radius
        x = range(0, int(rad + 1));
        for v in x:
            y = np.sqrt(pow(rad,2) - pow(v,2))
            # discount tiny computational error, then see if the answer is an integer
            if abs(y - round(y)) < pow(10, -10):
                nodes.append((v, int(round(y))))
                for i in [-v, v]:
                    if loc[0]+i < size[0] and loc[0]+i >= 0:
                        for j in [int(round(y)), -int(round(y))]:
                            #if it is within range
                            if loc[1]+j < size[1] and loc[1]+j >= 0:
                                #print(loc[0]+i, loc[1]+j)
                                #print(knowledge[loc[0]+i, loc[1]+j][0:3])
                                #print(sum(knowledge[loc[0]+i, loc[1]+j][0:3]))
                                #print(threshold)
                                #if sum(knowledge[loc[0]+i, loc[1]+j][0:3])/3 > threshold:
                                #    print("no", loc[0]+i, loc[1]+j)
                                if sum(knowledge[loc[0]+i, loc[1]+j][0:3])/3 < threshold:
                                    #   print(knowledge[loc[0]+i, loc[1]+j][0:3])
                                #    print("yes", loc[0]+i, loc[1]+j)
                                    return (loc[0]+i, loc[1]+j)

# includes a certainty matrix, which increases as repeat measurements are
# gathered over the same region, particularly if they are in accordance

# perhaps the BAyes approach could focus on regions, and group 'certain' regions 
# as one, to minimise computation

# we also want to render the certainty matrix
def Bayesian_2D(state, knowledge, stimulus, certainty, res_m):
    (state.x, state.y) = (int(state.x/res_m), int(state.y/res_m))
    (state.x, state.y) = find_nearest_empty(state, certainty, 0.6*255, 1)
    (state.x, state.y) = (state.x*res_m, state.y*res_m)
    #knowledge[state.x, state.y] = stimulus[int(state.x/res_m), int(state.y/res_m)]
    sd = 50
    measurement=[0,0,0,255]
    for i in [0, 1, 2]:
        #take measurement with some gaussian noise
        measurement[i] = np.random.normal(loc=stimulus[int(state.x/res_m), int(state.y/res_m)][i], scale = sd)
        if measurement[i] < 0:
            measurement[i] = 0
        if measurement[i] > 255:
            measurement[i] = 255
    #fill in the next available knowledge "slot"
    done = 0
    for i in range(0,res_m):
        for j in range(0,res_m):
            if knowledge[state.x+i, state.y+j][0] == -1 and done == 0:
                knowledge[state.x+i, state.y+j] = measurement
                done = 1
    #look in the neighbouring region
    for x in [int(state.x/res_m)-1, int(state.x/res_m), int(state.x/res_m)+1]:
        for y in [int(state.y/res_m)-1, int(state.y/res_m), int(state.y/res_m)+1]:
            space = np.shape(stimulus)
            # don't draw outside the area
            if x >= space[0] or y >= space[1] or x < 0 or y < 0:
                pass
            # update the certainty at the current location equally
            elif x == int(state.x/res_m) and y == int(state.y/res_m):
                #this should be a combination of integrating over the pdf and
                #using neighbouring cells to predict
                # - the strength of how well cells can predict their neighbours
                # can be trained over multiple data
                certainty[x, y] = [certainty[x, y][0]+255/(2*res_m*res_m),
                                   certainty[x, y][1]+255/(2*res_m*res_m),
                                   certainty[x, y][2]+255/(2*res_m*res_m),
                                   255]
            # update the neighbours with a green tint
            else:
                certainty[x, y] = [certainty[x, y][0],
                                   certainty[x, y][1]+255/(32*res_m*res_m),
                                   certainty[x, y][2],
                                   255]
    # also colour its neighbours (certainty)
    
    # Choose an action to minimise entropy of belief
    # beleif = P(O|h)
    # (intent parameter | history)
    # new belief = P(b'|y,a,b)
    # new belief = (b'|latest info,latest choice,old beleif)
    
    # What if my reward was guessing the stiffness/response, with diminishing
    # returns as the measurements are repeated?
    
    #  Certainty is some function of number of measurements and s.d. between them
    #  Perhaps if a ceiling number of measurements is ascertained, then the
    # current level of uncertainty can be compared to that. e.g. 20 readings
    # always results in a "true" value, so the s.d. can be calculated knowing
    # this
    #  Find the expected shape of the Bell-curve. Take few measurements, find
    # the likliest COM/mean from these measurements
    
    # Assuming a bell/gaussian curve:
    
    
#    certainty[int(state.x/res_m), int(state.y/res_m)] = 
    '''
    sigma = 10
    mean = 50
    # x axis is assumed density
    (1/(sigma*sqrt(2*pi)))*exp(-((x-mean)/sigma)^2/2)
    '''
    return (state, knowledge, certainty)

# subsidiary function
def Gaussian_dist(x):
    avg = 0 #avg doesn't need to be included - this is just for consistency with gaussian
    sd = 10
    prob_dens = (1/(sd*np.sqrt(2*np.pi)))*np.exp(-(((x-avg)/sd)**2)/2)
    return prob_dens

#  integrates over the tolerance to return the liklihood that the current value
# is within tolerance
def prob_from_gauss(tolerance, measurements):
    sd = 10
    cert = integrate.quad(lambda x: pow(Gaussian_dist(x),measurements)/ \
               (integrate.quad(lambda x: pow(Gaussian_dist(x),measurements),\
                               -5*sd, 5*sd)[0]), \
                   -tolerance, tolerance)
    return cert

######### 3D methods ####################################

#  This method rewards itself when it correctly predicts the reaction from an
# unexplored region of the stimulus.
#  The Bayesian/calculated matrix should have a finer level of detail
# (Nyquist-Shannon <f/2) than the resolution that is desired.
#  It explores to minimise entropy in its belief over the whole stimulus.
#  Belief function is initialised to have maximum entropy (min information).
#  Finishing condition can be one of:
    # minimum reward expected for any unknown region reaches threshold
    # the region with max entropy reaches a minimum


def Bayes_Approach(state, knowledge, stimulus):
    # selects and executes a movement, generating a new state, knowledge
    
    
    if Finishing_condition:
        return(state)
    else:
        Bayes_Approach(state, knowledge, stimulus)

# method moves a character in a worldstate. It has a choice of moving (1
# square) or "palpating". ------ This is similar to clicking in minesweeper(?)




































 