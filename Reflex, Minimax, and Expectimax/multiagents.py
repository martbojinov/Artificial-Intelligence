import random

from pacai.agents.base import BaseAgent
from pacai.agents.search.multiagent import MultiAgentSearchAgent
from pacai.core import distance

class ReflexAgent(BaseAgent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.
    You are welcome to change it in any way you see fit,
    so long as you don't touch the method headers.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        `ReflexAgent.getAction` chooses among the best options according to the evaluation function.

        Just like in the previous project, this method takes a
        `pacai.core.gamestate.AbstractGameState` and returns some value from
        `pacai.core.directions.Directions`.
        """

        # Collect legal moves.
        legalMoves = gameState.getLegalActions()
        if 'Stop' in legalMoves:    # remove 'Stop' move so that pacman always moving somwhere
            legalMoves.remove('Stop')
        # print(legalMoves)

        # Choose one of the best actions.
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best.

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current `pacai.bin.pacman.PacmanGameState`
        and an action, and returns a number, where higher numbers are better.
        Make sure to understand the range of different values before you combine them
        in your evaluation function.
        """

        successorGameState = currentGameState.generatePacmanSuccessor(action)

        # Useful information you can extract.
        newPosition = successorGameState.getPacmanPosition()
        oldFood = currentGameState.getFood()
        # newGhostPositions = successorGameState.getGhostPositions()
        # newGhostStates = successorGameState.getGhostStates()
        # newScaredTimes = [ghostState.getScaredTimer() for ghostState in newGhostStates]

        # Disance from pacman to food
        pacmanToFoodDistances = []
        for food in oldFood.asList():
            d = distance.manhattan(newPosition, food)
            pacmanToFoodDistances.append(d)

        finalScore = successorGameState.getScore() - min(pacmanToFoodDistances)

        return finalScore

class MinimaxAgent(MultiAgentSearchAgent):
    """
    A minimax agent.

    Here are some method calls that might be useful when implementing minimax.

    `pacai.core.gamestate.AbstractGameState.getNumAgents()`:
    Get the total number of agents in the game

    `pacai.core.gamestate.AbstractGameState.getLegalActions`:
    Returns a list of legal actions for an agent.
    Pacman is always at index 0, and ghosts are >= 1.

    `pacai.core.gamestate.AbstractGameState.generateSuccessor`:
    Get the successor game state after an agent takes an action.

    `pacai.core.directions.Directions.STOP`:
    The stop direction, which is always legal, but you may not want to include in your search.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the minimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    # see page 305 of textbook for psudocode for minmax()
    def value(self, gameState, currentDepth):
        (v, a) = self.maxValue(gameState, currentDepth)
        return a

    def maxValue(self, gameState, currentDepth):
        # terminal (including win/lose states)
        if currentDepth == self.getTreeDepth() or gameState.isWin() or gameState.isLose():
            return (self.getEvaluationFunction()(gameState), None)
        # node
        else:
            # Get legal moves for pacman
            legalMoves = gameState.getLegalActions()
            if 'Stop' in legalMoves:    # remove 'Stop' move so that pacman always moving somwhere
                legalMoves.remove('Stop')
            # Get successor states from legal moves
            successors = [gameState.generatePacmanSuccessor(action) for action in legalMoves]
            # Send successors to minValue func
            minPairs = [self.minValue(s, currentDepth) for s in successors]
            scores = list(list(zip(*minPairs))[0])    # convert pairs to usable score list
            # Get max of minValues
            bestScore = max(scores)
            bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
            chosenIndex = random.choice(bestIndices)  # Pick randomly among the best.
            return (scores[chosenIndex], legalMoves[chosenIndex])

    def minValue(self, gameState, currentDepth, agentNum = 1):
        # terminal (including win/lose states)
        if currentDepth == self.getTreeDepth() or gameState.isWin() or gameState.isLose():
            return (self.getEvaluationFunction()(gameState), None)
        # node
        else:
            # Get legal moves of ghost agent
            legalMoves = gameState.getLegalActions(agentNum)
            if 'Stop' in legalMoves:    # remove 'Stop' move so that pacman always moving somwhere
                legalMoves.remove('Stop')
            # Get successor states from legal moves
            successors = [gameState.generateSuccessor(agentNum, action) for action in legalMoves]
            # Send successors to minValue or maxValue func (multiple minValue layers, 1 per ghost)
            minPairs = []
            if agentNum == gameState.getNumAgents() - 1:    # all ghosts done
                minPairs = [self.maxValue(s, currentDepth + 1) for s in successors]
            else:                                           # more ghosts to be done
                minPairs = [self.minValue(s, currentDepth, agentNum + 1) for s in successors]
            scores = list(list(zip(*minPairs))[0])    # convert pairs to usable score list
            # Get min of maxValues (or previous layers of minValue)
            bestScore = min(scores)
            bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
            chosenIndex = random.choice(bestIndices)  # Pick randomly among the best.
            return (scores[chosenIndex], legalMoves[chosenIndex])

    def getAction(self, gameState):
        a = self.value(gameState, 0)

        return a

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    A minimax agent with alpha-beta pruning.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the minimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    # see page 311 of textbook for psudocode for alphaBeta()
    def value(self, gameState, currentDepth):
        (v, a) = self.maxValue(gameState, currentDepth)
        return a

    def maxValue(self, gameState, currentDepth, alpha = -999999, beta = 999999):
        # terminal (including win/lose states)
        if currentDepth == self.getTreeDepth() or gameState.isWin() or gameState.isLose():
            return (self.getEvaluationFunction()(gameState), None)
        # node
        else:
            # Get legal moves for pacman
            legalMoves = gameState.getLegalActions()
            if 'Stop' in legalMoves:    # remove 'Stop' move so that pacman always moving somwhere
                legalMoves.remove('Stop')
            # Get successor states from legal moves
            successors = [gameState.generatePacmanSuccessor(action) for action in legalMoves]
            # Send successors to minValue func
            minPairs = []
            for s in successors:
                (v2, a2) = self.minValue(s, currentDepth, alpha=alpha, beta=beta)
                minPairs.append((v2, a2))
                alpha = max(alpha, v2)
                if v2 >= beta:
                    index = successors.index(s)
                    return (v2, legalMoves[index])
            scores = list(list(zip(*minPairs))[0])    # convert pairs to usable score list
            # Get max of minValues
            bestScore = max(scores)
            bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
            chosenIndex = random.choice(bestIndices)  # Pick randomly among the best.
            return (scores[chosenIndex], legalMoves[chosenIndex])

    def minValue(self, gameState, currentDepth, agentNum = 1, alpha = -999999, beta = 999999):
        # terminal (including win/lose states)
        if currentDepth == self.getTreeDepth() or gameState.isWin() or gameState.isLose():
            return (self.getEvaluationFunction()(gameState), None)
        # node
        else:
            # Get legal moves of ghost agent
            legalMoves = gameState.getLegalActions(agentNum)
            if 'Stop' in legalMoves:    # remove 'Stop' move so that pacman always moving somwhere
                legalMoves.remove('Stop')
            # Get successor states from legal moves
            successors = [gameState.generateSuccessor(agentNum, action) for action in legalMoves]
            # Send successors to minValue or maxValue func (multiple minValue layers, 1 per ghost)
            minPairs = []
            if agentNum == gameState.getNumAgents() - 1:    # all ghosts done
                for s in successors:
                    (v2, a2) = self.maxValue(s, currentDepth + 1, alpha=alpha, beta=beta)
                    minPairs.append((v2, a2))
                    beta = min(beta, v2)
                    if v2 >= alpha:
                        index = successors.index(s)
                        return (v2, legalMoves[index])
            else:                                           # more ghosts to be done
                for s in successors:
                    (v2, a2) = self.minValue(s, currentDepth, agentNum + 1, alpha=alpha, beta=beta)
                    minPairs.append((v2, a2))
                    beta = min(beta, v2)
                    if v2 >= alpha:
                        index = successors.index(s)
                        return (v2, legalMoves[index])
            scores = list(list(zip(*minPairs))[0])    # convert pairs to usable score list
            # Get min of maxValues (or previous layers of minValue)
            bestScore = min(scores)
            bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
            chosenIndex = random.choice(bestIndices)  # Pick randomly among the best.
            return (scores[chosenIndex], legalMoves[chosenIndex])

    def getAction(self, gameState):
        a = self.value(gameState, 0)

        return a

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    An expectimax agent.

    All ghosts should be modeled as choosing uniformly at random from their legal moves.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the expectimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    # see page 305 of textbook for psudocode for minmax()
    def value(self, gameState, currentDepth):
        (v, a) = self.maxValue(gameState, currentDepth)
        return a

    def maxValue(self, gameState, currentDepth):
        # terminal (including win/lose states)
        if currentDepth == self.getTreeDepth() or gameState.isWin() or gameState.isLose():
            return (self.getEvaluationFunction()(gameState), None)
        # node
        else:
            # Get legal moves for pacman
            legalMoves = gameState.getLegalActions()
            if 'Stop' in legalMoves:    # remove 'Stop' move so that pacman always moving somwhere
                legalMoves.remove('Stop')
            # Get successor states from legal moves
            successors = [gameState.generatePacmanSuccessor(action) for action in legalMoves]
            # Send successors to expValue func
            minPairs = [self.expValue(s, currentDepth) for s in successors]
            scores = list(list(zip(*minPairs))[0])    # convert pairs to usable score list
            # Get max of expValues
            bestScore = max(scores)
            bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
            chosenIndex = random.choice(bestIndices)  # Pick randomly among the best.
            return (scores[chosenIndex], legalMoves[chosenIndex])

    def expValue(self, gameState, currentDepth, agentNum = 1):
        # terminal (including win/lose states)
        if currentDepth == self.getTreeDepth() or gameState.isWin() or gameState.isLose():
            return (self.getEvaluationFunction()(gameState), None)
        # node
        else:
            # Get legal moves of ghost agent
            legalMoves = gameState.getLegalActions(agentNum)
            if 'Stop' in legalMoves:    # remove 'Stop' move so that pacman always moving somwhere
                legalMoves.remove('Stop')
            # Get successor states from legal moves
            successors = [gameState.generateSuccessor(agentNum, action) for action in legalMoves]
            # Send successors to expValue or maxValue func (multiple minValue layers, 1 per ghost)
            minPairs = []
            if agentNum == gameState.getNumAgents() - 1:    # all ghosts done
                minPairs = [self.maxValue(s, currentDepth + 1) for s in successors]
            else:                                           # more ghosts to be done
                minPairs = [self.expValue(s, currentDepth, agentNum + 1) for s in successors]
            scores = list(list(zip(*minPairs))[0])    # convert pairs to usable score list
            # Get expectedUtility of maxValues (or previous layers of expValue)
            expectedUtility = sum(scores) / len(scores)
            # Current choice of move action according to minimax (not important)
            bestScore = min(scores)
            bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
            chosenIndex = random.choice(bestIndices)  # Pick randomly among the best.
            return (expectedUtility, legalMoves[chosenIndex])

    def getAction(self, gameState):
        a = self.value(gameState, 0)

        return a

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable evaluation function.

    DESCRIPTION: <write something here so we know what you did>
    In my betterEvaluationFunction, I tracked the number of pelletes on the board,
    the distance to the farthest pellet, and the distance to the closest ghost.
    For each of these calculations, the reciprocal is taken and multiplied by a weight.
    The weights are:
        food pellets:                       100
        distance to farthest food pellet:   100
        distance to closest ghost:          150
    """
    # game state information
    food = currentGameState.getFood().asList()
    pacmanPos = currentGameState.getPacmanPosition()
    ghostsPos = currentGameState.getGhostPositions()

    # track the number of food pellets left on the board
    numFood = 0
    if len(food) > 0:   # edge case for 0 pellets
        numFood = 100 * (1 / len(food))

    # track the distance to the farthest food pellet on the board
    pacmanToFoodDistances = []
    for f in food:
        d = distance.manhattan(pacmanPos, f)
        pacmanToFoodDistances.append(d)
    foodDist = 0
    if len(pacmanToFoodDistances) > 0:  # edge case for 0 pellets
        foodDist = 100 * (1 / max(pacmanToFoodDistances))

    # track the disance from pacman to the closes ghost
    pacmanToGhostDistances = []
    for g in ghostsPos:
        d = distance.manhattan(pacmanPos, g)
        if d != 0:      # edge case where distance = 0; leads to div by 0 error
            pacmanToGhostDistances.append(d)
    ghostDist = 0
    if len(pacmanToGhostDistances) > 0:  # edge case for 0 pellets
        ghostDist = 150 * (1 / min(pacmanToGhostDistances))

    return currentGameState.getScore() + numFood + foodDist + ghostDist

class ContestAgent(MultiAgentSearchAgent):
    """
    Your agent for the mini-contest.

    You can use any method you want and search to any depth you want.
    Just remember that the mini-contest is timed, so you have to trade off speed and computation.

    Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
    just make a beeline straight towards Pacman (or away if they're scared!)

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)
