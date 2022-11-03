# mlLearningAgents.py
# parsons/27-mar-2017
#
# A stub for a reinforcement learning agent to work with the Pacman
# piece of the Berkeley AI project:
#
# http://ai.berkeley.edu/reinforcement.html
#
# As required by the licensing agreement for the PacMan AI we have:
#
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

# This template was originally adapted to KCL by Simon Parsons, but then
# revised and updated to Py3 for the 2022 course by Dylan Cope and Lin Li

from __future__ import absolute_import
from __future__ import print_function

import random

from pacman import Directions, GameState
from pacman_utils.game import Agent
from pacman_utils import util
from typing import List


class GameStateFeatures:
    """
    Wrapper class around a game state where you can extract
    useful information for your Q-learning algorithm

    WARNING: We will use this class to test your code, but the functionality
    of this class will not be tested itself
    """

    def __init__(self, state: GameState):
        """
        Args:
            state: A given game state object
        """

        "*** YOUR CODE HERE ***"
        # Set the pacamn position
        self.state = state

        # reference: Hruthik Ketepalli. kcl forum https://keats.kcl.ac.uk/mod/forum/discuss.php?d=485452
    def __eq__(self, other):
        """
        Allows two states to be compared.
        """
        return self.state.__eq__(other.state)

     # reference: Hruthik Ketepalli. kcl forum https://keats.kcl.ac.uk/mod/forum/discuss.php?d=485452
    def __hash__(self):
        """
        Allows states to be keys of dictionaries.
        """
        return hash(self.state)


class QLearnAgent(Agent):

    def __init__(self,
                 alpha: float = 0.2,
                 epsilon: float = 0.05,
                 gamma: float = 0.8,
                 maxAttempts: int = 30,
                 numTraining: int = 10):
        """
        These values are either passed from the command line (using -a alpha=0.5,...)
        or are set to the default values above.

        The given hyperparameters are suggestions and are not necessarily optimal
        so feel free to experiment with them.

        Args:
            alpha: learning rate
            epsilon: exploration rate
            gamma: discount factor
            maxAttempts: How many times to try each action in each state
            numTraining: number of training episodes
        """
        super().__init__()
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.gamma = float(gamma)
        self.maxAttempts = int(maxAttempts)
        self.numTraining = int(numTraining)
        # Count the number of games we have played
        self.episodesSoFar = 0

        # Using Dictionary to represent Q-table to store all Q-values. e.g. {(state,action):(qValue, times)}
        self.qTable = {}

        # Record the previous game states
        self.lastState = []
        # Record the previous actions
        self.lastAction = []

    # Accessor functions for the variable episodesSoFar controlling learning

    def incrementEpisodesSoFar(self):
        self.episodesSoFar += 1

    def getEpisodesSoFar(self):
        return self.episodesSoFar

    def getNumTraining(self):
        return self.numTraining

    # Accessor functions for parameters
    def setEpsilon(self, value: float):
        self.epsilon = value

    def getAlpha(self) -> float:
        return self.alpha

    def setAlpha(self, value: float):
        self.alpha = value

    def getGamma(self) -> float:
        return self.gamma

    def getMaxAttempts(self) -> int:
        return self.maxAttempts

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    @staticmethod
    def computeReward(startState: GameState,
                      endState: GameState) -> float:
        """
        Args:
            startState: A starting state
            endState: A resulting state

        Returns:
            The reward assigned for the given trajectory
        """
        "*** YOUR CODE HERE ***"
        # Reward depends on the change between two states' scores
        startScore = startState.getScore()
        endScore = endState.getScore()

        return endScore - startScore

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getQValue(self,
                  state: GameStateFeatures,
                  action: Directions) -> float:
        """
        Args:
            state: A given state
            action: Proposed action to take

        Returns:
            Q(state, action)
        """
        "*** YOUR CODE HERE ***"
        if (state, action) in self.qTable:
            # Get the action dictionary
            # which relates to the specific position
            return self.qTable[(state, action)][0]
        else:
            # Return 0 if the position is never took
            return 0.0

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def maxQValue(self, state: GameStateFeatures) -> float:
        """
        Args:
            state: The given state

        Returns:
            q_value: the maximum estimated Q-value attainable from the state
        """
        "*** YOUR CODE HERE ***"
        qList = []

        # Search for the stored Q-values
        for key in self.qTable:
            if key[0] == state:
                qValue = self.qTable[key][0]
                qList.append(qValue)

        # Return 0 if there is no Q-values
        # Else return thr max Q-values
        if len(qList) == 0:
            return 0
        else:
            return max(qList)

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def learn(self,
              state: GameStateFeatures,
              action: Directions,
              reward: float,
              nextState: GameStateFeatures):
        """
        Performs a Q-learning update

        Args:
            state: the initial state
            action: the action that was took
            nextState: the resulting state
            reward: the reward received on this trajectory
        """
        "*** YOUR CODE HERE ***"
        # Calculate the Q-value
        q = self.getQValue(state, action)
        nextMaxQ = self.maxQValue(nextState)

        # Exploration
        #counts = self.qTable[(state, action)][1]

        # Q-learning
        qValue = q + self.alpha * (reward + self.gamma * nextMaxQ - q)

        # Update the Q-value with specific action and position in the Q-table
        if(state, action) in self.qTable:
            times = self.qTable[(state, action)][1]
            self.qTable[(state, action)] = (qValue, times)
        else:
            self.qTable[(state, action)] = (0, 0)

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature

    def updateCount(self,
                    state: GameStateFeatures,
                    action: Directions):
        """
        Updates the stored visitation counts.

        Args:
            state: Starting state
            action: Action taken
        """
        "*** YOUR CODE HERE ***"
        stWithAc = (state, action)
        if stWithAc in self.qTable:
            qValue = self.qTable[(stWithAc)][0]
            times = self.qTable[(stWithAc)][1]
            self.qTable[stWithAc] = (qValue, times + 1)
        else:
            self.qTable[(state, action)] = (0, 1)

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getCount(self,
                 state: GameStateFeatures,
                 action: Directions) -> int:
        """
        Args:
            state: Starting state
            action: Action taken

        Returns:
            Number of times that the action has been taken in a given state
        """
        "*** YOUR CODE HERE ***"
        return self.qTable[(state, action)][1]

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature

    def explorationFn(self,
                      utility: float,
                      counts: int) -> float:
        """
        Computes exploration function.
        Return a value based on the counts

        HINT: Do a greed-pick or a least-pick

        Args:
            utility: expected utility for taking some action a in some given state s
            counts: counts for having taken visited

        Returns:
            The exploration value
        """
        "*** YOUR CODE HERE ***"
        # decrease epsilon during the trianing
        # ep = 1 - self.getEpisodesSoFar()*1.0/self.getNumTraining()
        # util.raiseNotDefined()

    # Get the best action depends to the max Q-values
    def getBestAction(self, stateFeature: GameStateFeatures, legalActions: List) -> Directions:
        actionWithQ = {}
        for action in legalActions:
            actionWithQ[action] = self.getQValue(stateFeature, action)

        # get action with max Q-value
        bestAction = max(actionWithQ, key=actionWithQ.get)

        return bestAction

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getAction(self, state: GameState) -> Directions:
        """
        Choose an action to take to maximise reward while
        balancing gathering data for learning

        If you wish to use epsilon-greedy exploration, implement it in this method.
        HINT: look at pacman_utils.util.flipCoin

        Args:
            state: the current state

        Returns:
            The action to take
        """
        # The data we have about the state of the game
        legal = state.getLegalPacmanActions()
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)

        # logging to help you understand the inputs, feel free to remove
        # print("Legal moves: ", legal)
        # print("Pacman position: ", currentPacmanPosition)
        # print("Ghost positions:", enemyPosition)
        # print("Food locations: ")
        # print(state.getFood())
        # print("Score: ", state.getScore())

        stateFeatures = GameStateFeatures(state)

        # Calculate reward between last state and current state
        if len(self.lastState) > 0:
            lastState = self.lastState[-1]
            lastAction = self.lastAction[-1]
            curReward = self.computeReward(lastState, state)

            # Update Q-Value
            self.learn(GameStateFeatures(lastState),
                       lastAction, curReward, stateFeatures)

        # e-greed-pick
        if util.flipCoin(self.epsilon):
            action = random.choice(legal)
        else:
            action = self.getBestAction(stateFeatures, legal)

        # update counts
        self.updateCount(stateFeatures, action)

        # Record the last state
        self.lastState.append(state)
        self.lastAction.append(action)

        # Now pick what action to take.
        return action

    def final(self, state: GameState):
        """
        Handle the end of episodes.
        This is called by the game after a win or a loss.

        Args:
            state: the final game state
        """
        print(f"Game {self.getEpisodesSoFar()} just ended!")
        print("final score", state.getScore())

        stateFeatures = GameStateFeatures(state)

        # Update final state's Q-value
        lastState = self.lastState[-1]
        lastAction = self.lastAction[-1]
        curReward = self.computeReward(lastState, state)
        self.learn(GameStateFeatures(lastState),
                   lastAction, curReward, stateFeatures)

        # Reset lists
        self.lastAction = []
        self.lastState = []

        # Keep track of the number of games played, and set learning
        # parameters to zero when we are done with the pre-set number
        # of training episodes
        self.incrementEpisodesSoFar()
        if self.getEpisodesSoFar() == self.getNumTraining():
            msg = 'Training Done (turning off epsilon and alpha)'
            print('%s\n%s' % (msg, '-' * len(msg)))
            self.setAlpha(0)
            self.setEpsilon(0)
