"""
Author: Darrien Mckenzie

File: HW4_Q1

Purpose: To be used for RL Homework #4, Question 1
"""
from gridworld import Gridworld
import numpy as np

def initialize_Q(states, actions, initial_value=1):
    Q = [[]] * states
    for state in range(states):
        Q[state] = [initial_value]*actions
    #print(Q)
    
    return Q

def Q_learning(MDP, iterations, alpha, discounting, epsilon):
    Q = initialize_Q(MDP.total_states, len(MDP.action_space), discounting)
    
    best_reward = None
    iterations_to_achieve_best = None
    for episode in range(iterations):
        state = MDP.reset()
        done = False
        epsiode_reward = 0
        steps = 0
        while not done:
            if np.random.random() < epsilon:
                action = np.argmax(np.asarray(Q[state]))
            else:
                action_index = np.random.randint(0,len(MDP.action_space))
                action = MDP.action_space[action_index]
            
            next_state, reward, done = MDP.step(action) #getting the next state, and reward recieved from taking action in current state
            Q[state][action] = Q[state][action] + alpha*(reward + discounting*np.max(np.asarray(Q[next_state])) - Q[state][action])
            
            if steps > max_steps:
                done = True
            
            epsiode_reward += reward
            steps += 1
        
        if best_reward is not None:
            if epsiode_reward > best_reward:
                best_reward = epsiode_reward
                iterations_to_achieve_best = episode
                print("NEW BEST REWARD: ", best_reward)
                print("ACHIEVED IN: " + str(iterations_to_achieve_best) + " ITERATIONS.")
        
        

if __name__ == "__main__":
    gridworld_transitions = {
    'S': ["START",-1],
    'N': ["NORMAL",-1],
    'G': ["TERMINATE",-1]
    }
    
    gridworld_layout = [['S', 'N', 'N', 'N'],
                        ['N', 'N', 'N', 'N'],
                        ['N', 'N', 'N', 'N'],
                        ['N', 'N', 'N', 'G']]
    
    off_reward = -1 #if agent tries to 'walk off' the gridworld, they will still recieve a reward of -1, and stay in the same state
    
    #all cells are numbered left to right, starting from first row
    starting_position = 0 #start at top left cell (cell 0)
    
    gridworld = Gridworld(gridworld_layout,gridworld_transitions,starting_position,off_reward) #initialize MDP
    Q = initialize_Q(gridworld.total_states, len(gridworld.action_space))
    