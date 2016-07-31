import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import pandas as pd

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        # sets self.env = env, state = None, next_waypoint = None,
        # and a default color
        super(LearningAgent, self).__init__(env)  
        self.color = 'red'  # override color
        # simple route planner to get next_waypoint
        self.planner = RoutePlanner(self.env, self)  
        # TODO: Initialize any additional variables here
        # Define the q matrix
        self.Q = pd.DataFrame(columns=['None', 'forward', 'left', 'right'])
        # Define Gamma of the Q learning algorithm
        self.gamma = 0.2
        # Define alpha of the Q learning algorith
        self.alpha = 0.8
        # Define the initialisation of Q
        self.q_init = 4.0

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required

    def make_state(self, inputs):
        """
        A function to make a String form of the state, based on the inputs
        the agent senses from the environment.

        Args:
            inputs: The inputs that the agent senses from the environment.

        Returns:
            state: a String representing the State in which the agent is found.
        """
        light = inputs['light']
        left = inputs['left']
        right = inputs['right']
        state = "{} {} {} {}".format(light, left, right, self.next_waypoint)
        return state

    def max_q(self, next_state):
        """
        A function to get the maximum q given the next state
        """
        maximum = 0
        if next_state in self.Q.index:
            maximum = self.Q.loc[next_state].max()
        else:
            maximum = self.q_init
        return maximum


    def learn(self, action, reward):
        """
        A function to learn the Q matrix.
        """
        inputs = self.env.sense(self)
        next_state = self.make_state(inputs)
        if self.state not in self.Q.index:
            self.Q.loc[self.state] = [self.q_init]*4
            return
        new_q = self.alpha*(reward + self.gamma * self.max_q(next_state))
        old_q = (1-self.alpha)*(self.Q.loc[self.state, str(action)])
        self.Q.loc[self.state, str(action)] = old_q + new_q

    def get_action(self):
        """
        A function to get the best action given a state
        """
        if self.state in self.Q.index:
            best_action = self.Q.loc[self.state].argmax()
        else:
            best_action = random.choice([None, 'forward', 'left', 'right'])
        if best_action == 'None':
            best_action = None
        return best_action

    def update(self, t):
        # Gather inputs
        # From route planner, also displayed by simulator
        self.next_waypoint = self.planner.next_waypoint()
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)


        # TODO: Update state
        self.state = "{} {} {} {}".format(inputs['light'], inputs['left'], 
                                       inputs['right'], self.next_waypoint)
        
        # TODO: Select action according to your policy
        action = self.get_action()

        # Execute action and get reward
        reward = self.env.act(self, action)

        # TODO: Learn policy based on state, action, reward
        self.learn(action, reward)

        # print self.Q

        # [debug]
        print ("LearningAgent.update(): deadline = {}, inputs = {}, action"
        " = {}, reward = {}").format(deadline, inputs, action, reward)


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent

    # create environment (also adds some dummy traffic)
    e = Environment()
    # create agent
    a = e.create_agent(LearningAgent)
    # specify agent to track
    e.set_primary_agent(a, enforce_deadline=True)
    # NOTE: You can set enforce_deadline=False while debugging to
    # allow longer trials

    # Now simulate it
    # create simulator (uses pygame when display=True, if available)
    sim = Simulator(e, update_delay=0.3, display=True)  
    #NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit 
    # Ctrl+C on the command-line


if __name__ == '__main__':
    run()
