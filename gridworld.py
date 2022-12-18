"""
Author: Darrien McKenzie

File: gridworld.py

Description: This file contains a class that allows for the implementation
of any gridworld environnment. This class was previously written for self-studying
RL endeavors.
"""
class Gridworld():
    def __init__(self,layout,transitions,start_state=0,off_reward=0):
        super().__init__()
        self.action_space = ['UP','RIGHT','DOWN','LEFT']
        self.off_reward = off_reward
                
        self.rows = len(layout)
        self.columns = len(layout[0])
        
        state_num = 0
        self.state_space = {}
        for row in range(self.rows):
            for col in range(self.columns):
                self.state_space[state_num] = (layout[row][col], (row,col))
                state_num += 1
        self.total_states = state_num
        
        self.transitions = transitions
        self.transition_space = {
        "START": self.normal,
        'NORMAL': self.normal,
        'TERMINATE': self.terminate,
        "BLOCK": self.block,
        "OFF": self.off
        }
        
        self.start_state = start_state
        self.state = self.start_state
        self.next_state = self.state

        self.reward = off_reward
        self.done = False
        
    def reset(self):
        self.state = self.start_state
        self.done = False
        
        return self.state
    
    def terminate(self):
        self.done = True
        return self.next_state
    
    def normal(self):
        return self.next_state
    
    def block(self):
        return self.state
    
    def off(self):
        return self.state
    
    def step(self,action):
        
        if action == 'UP':
            self.next_state = self.state - self.columns
        elif action == 'RIGHT':
            self.next_state = self.state + 1
        elif action == 'DOWN':
            self.next_state = self.state + self.columns
        elif action=='LEFT':
            self.next_state = self.state - 1
        
        
        if ((self.next_state >= self.columns*self.rows) or (self.next_state < 0)):
            self.next_state = self.off()
            self.reward = self.off_reward
        else:
            self.next_state = self.transition_space[self.transitions[self.state_space[self.next_state][0]][0]]()
            self.reward = self.transitions[self.state_space[self.next_state][0]][1]
        
        print("CURRENT STATE = " + str(self.state))
        self.state = self.next_state
        print("NEXT STATE = " + str(self.state))
        return self.state, self.reward, self.done
    
    def render(self):
        line = ""
        inc = 0
        for row in range(self.rows):
            for col in range(self.columns):
                #print(type(self.state_space[self.state][1][0]))
                if row == self.state_space[self.state][1][0] and col == self.state_space[self.state][1][1]:
                    line = line + 'C' + " "
                else:
                    line = line + str(self.state_space[inc][0]) + " "
                inc = inc + 1
            print(line)
            line = ""
        print()      