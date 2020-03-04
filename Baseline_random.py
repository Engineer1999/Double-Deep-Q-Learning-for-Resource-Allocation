from __future__ import division, print_function
import numpy as np 
from Environment import *
import matplotlib.pyplot as plt

# This py file using the random algorithm.

def main():
    up_lanes = [3.5/2,3.5/2 + 3.5,250+3.5/2, 250+3.5+3.5/2, 500+3.5/2, 500+3.5+3.5/2]
    down_lanes = [250-3.5-3.5/2,250-3.5/2,500-3.5-3.5/2,500-3.5/2,750-3.5-3.5/2,750-3.5/2]
    left_lanes = [3.5/2,3.5/2 + 3.5,433+3.5/2, 433+3.5+3.5/2, 866+3.5/2, 866+3.5+3.5/2]
    right_lanes = [433-3.5-3.5/2,433-3.5/2,866-3.5-3.5/2,866-3.5/2,1299-3.5-3.5/2,1299-3.5/2]
    width = 750
    height = 1299
    n = 40
    Env = Environ(down_lanes,up_lanes,left_lanes,right_lanes, width, height)
    number_of_game = 50
    n_step = 100
    V2I_Rate_List = np.zeros([number_of_game, n_step])
    Fail_Percent = np.zeros([number_of_game, n_step])
    for game_idx in range(number_of_game):
        print (game_idx)
        Env.new_random_game(n)
        for i in range(n_step):
            #print(i)
            actions = np.random.randint(0,20,[n,3])
            power_selection = np.zeros(actions.shape, dtype = 'int')
            actions = np.concatenate((actions[..., np.newaxis],power_selection[...,np.newaxis]), axis = 2)
            reward, percent = Env.act(actions)
            V2I_Rate_List[game_idx, i] = np.sum(reward)
            Fail_Percent[game_idx, i] = percent
        print(np.sum(reward))
        print ('percentage here is ', percent)
    print ('The number of vehicles is ', n)
    print ('mean of V2I rate is that ', np.mean(V2I_Rate_List))
    print ('mean of percent is ', np.mean(Fail_Percent[:,-1]))

main()
