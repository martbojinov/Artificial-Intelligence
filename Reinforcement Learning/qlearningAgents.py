import random

from pacai.agents.learning.reinforcement import ReinforcementAgent
from pacai.util import reflection
from pacai.util import probability

class QLearningAgent(ReinforcementAgent):
    """
    A Q-Learning agent.

    Some functions that may be useful:

    `pacai.agents.learning.reinforcement.ReinforcementAgent.getAlpha`:
    Get the learning rate.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.getDiscountRate`:
    Get the discount rate.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.getEpsilon`:
    Get the exploration probability.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.getLegalActions`:
    Get the legal actions for a reinforcement agent.

    `pacai.util.probability.flipCoin`:
    Flip a coin (get a binary value) with some probability.

    `random.choice`:
    Pick randomly from a list.

    Additional methods to implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Compute the action to take in the current state.
    With probability `pacai.agents.learning.reinforcement.ReinforcementAgent.getEpsilon`,
    we should take a random action and take the best policy action otherwise.
    Note that if there are no legal actions, which is the case at the terminal state,
    you should choose None as the action.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.update`:
    The parent class calls this to observe a state transition and reward.
    You should do your Q-Value update here.
    Note that you should never call this function, it will be called on your behalf.

    DESCRIPTION:
    - In init(), there is a qvalue dictionary.
    - When the update function is called, qvalues are evaluated and added to the qvalue dictionary.
    - getQValue() simply grabs whatever value it is given from the dictionary (and returns 0
      if it cannot get anything).
    - getValue() and getPolicy() work the same way in that they get the highest-valued qValue
      from the current state but return their respective parts of the qValue; if there is a tie,
      it is broken randomly.
    - getAction() works similar to getAction from valueIterationAgent, in that it gets the next
      action based on the policy. It differs in that there is chance for random exploration.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

        # You can initialize Q-values here.
        # dictionary format is --> (state, action) : reward
        self.qvalues = {}  # A dictionary which holds the q-values for each state.

    def update(self, state, action, nextState, reward):
        """
        This class will call this function after observing a transition and reward.
        This function updates the qvalues dictionary with new rewards as they are observed.
        """

        oldValue = self.getQValue(state, action)
        learningRate = self.getAlpha()
        discountRate = self.getDiscountRate()
        optimalFutureValue = self.getValue(nextState)
        temporalDifference = reward + discountRate * optimalFutureValue
        newValue = ((1 - learningRate) * oldValue) + (learningRate * temporalDifference)

        self.qvalues[(state, action)] = newValue

    def getQValue(self, state, action):
        """
        Get the Q-Value for a `pacai.core.gamestate.AbstractGameState`
        and `pacai.core.directions.Directions`.
        Should return 0.0 if the (state, action) pair has never been seen.
        """

        return self.qvalues.get((state, action), 0.0)

    def getValue(self, state):
        """
        Return the value of the best action in a state.
        I.E., the value of the action that solves: `max_action Q(state, action)`.
        Where the max is over legal actions.
        Note that if there are no legal actions, which is the case at the terminal state,
        you should return a value of 0.0.

        This method pairs with `QLearningAgent.getPolicy`,
        which returns the actual best action.
        Whereas this method returns the value of the best action.
        """
        actions = self.getLegalActions(state)

        if len(actions) == 0:   # terminal state, return 0.0
            return 0.0

        # non-terminal state
        legalQValues = []
        for a in actions:   # get all aValues of state and put in a list
            legalQValues.append(self.getQValue(state, a))

        # return the max qValue value. if more than one, choose at random.
        bestQValue = max(legalQValues)
        bestIndices = [i for i in range(len(legalQValues)) if legalQValues[i] == bestQValue]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best.

        return legalQValues[chosenIndex]

    def getPolicy(self, state):
        """
        Return the best action in a state.
        I.E., the action that solves: `max_action Q(state, action)`.
        Where the max is over legal actions.
        Note that if there are no legal actions, which is the case at the terminal state,
        you should return a value of None.

        This method pairs with `QLearningAgent.getValue`,
        which returns the value of the best action.
        Whereas this method returns the best action itself.
        """
        actions = self.getLegalActions(state)

        if len(actions) == 0:   # terminal state, return None
            return None

        # non-terminal state
        legalQValues = []
        for a in actions:   # get all qValues of state and put in a list
            legalQValues.append(self.getQValue(state, a))

        # return the max qValue action. if more than one, choose at random.
        bestQValue = max(legalQValues)
        bestIndices = [i for i in range(len(legalQValues)) if legalQValues[i] == bestQValue]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best.

        return actions[chosenIndex]

    def getAction(self, state):
        """
        Returns the policy at the state or choose randomly of chance epsilon.
        """
        epsilon = self.getEpsilon()
        if probability.flipCoin(epsilon):   # explore randomly
            actions = self.getLegalActions(state)
            if len(actions) == 0:               # terminal state, return None
                return None
            return random.choice(actions)
        else:                               # explore by policy
            return self.getPolicy(state)

class PacmanQAgent(QLearningAgent):
    """
    Exactly the same as `QLearningAgent`, but with different default parameters.
    """

    def __init__(self, index, epsilon = 0.05, gamma = 0.8, alpha = 0.2, numTraining = 0, **kwargs):
        kwargs['epsilon'] = epsilon
        kwargs['gamma'] = gamma
        kwargs['alpha'] = alpha
        kwargs['numTraining'] = numTraining

        super().__init__(index, **kwargs)

    def getAction(self, state):
        """
        Simply calls the super getAction method and then informs the parent of an action for Pacman.
        Do not change or remove this method.
        """

        action = super().getAction(state)
        self.doAction(state, action)

        return action

class ApproximateQAgent(PacmanQAgent):
    """
    An approximate Q-learning agent.

    You should only have to overwrite `QLearningAgent.getQValue`
    and `pacai.agents.learning.reinforcement.ReinforcementAgent.update`.
    All other `QLearningAgent` functions should work as is.

    Additional methods to implement:

    `QLearningAgent.getQValue`:
    Should return `Q(state, action) = w * featureVector`,
    where `*` is the dotProduct operator.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.update`:
    Should update your weights based on transition.

    DESCRIPTION: Similar to previous qAgent but using the new system of features and weights.
    - getQValue() returns the current qValue using current weights, values at a (state, action)
    - update() gets qValues of states and uses them to update the weights dictionary.
    """

    def __init__(self, index,
            extractor = 'pacai.core.featureExtractors.IdentityExtractor', **kwargs):
        super().__init__(index, **kwargs)
        self.featExtractor = reflection.qualifiedImport(extractor)

        # You might want to initialize weights here.
        # dictionary format is --> feature : weight
        self.weights = {}  # A dictionary which holds the weights for each feature.

    def update(self, state, action, nextState, reward):
        """
        This class will call this function after observing a transition and reward.
        This function updates the qvalues dictionary with new rewards as they are observed.
        """
        features = self.featExtractor.getFeatures(self, state, action)

        for f, fv in features.items():
            oldWeight = self.weights.get(f, 0.0)
            learningRate = self.getAlpha()
            discountRate = self.getDiscountRate()
            optimalFutureValue = self.getValue(nextState)
            temporalDifference = reward + (discountRate * optimalFutureValue)
            newWeight = oldWeight + (learningRate * temporalDifference)
            self.weights[f] = newWeight

    def getQValue(self, state, action):
        """
        Get the Q-Value for a `pacai.core.gamestate.AbstractGameState`
        and `pacai.core.directions.Directions`.
        Should return 0.0 if the (state, action) pair has never been seen.
        """
        sum = 0.0
        features = self.featExtractor.getFeatures(self, state, action)
        for f, fv in features.items():  # dictionary of features : values
            sum += self.weights.get(f, 0.0) * fv    # w * fv

        return sum

    def final(self, state):
        """
        Called at the end of each game.
        """

        # Call the super-class final method.
        super().final(state)

        # Did we finish training?
        if self.episodesSoFar == self.numTraining:
            # You might want to print your weights here for debugging.
            # *** Your Code Here ***
            pass
