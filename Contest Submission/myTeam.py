from pacai.agents.capture.capture import CaptureAgent
import random

def createTeam(firstIndex, secondIndex, isRed,
        first = 'pacai.student.minmax.MinMaxAgent',
        second = 'pacai.student.minmax.MinMaxAgent'):
    """
    This function should return a list of two agents that will form the capture team,
    initialized using firstIndex and secondIndex as their agent indexed.
    isRed is True if the red team is being created,
    and will be False if the blue team is being created.
    """

    firstAgent = MinMaxAgent(firstIndex)
    secondAgent = MinMaxAgent(secondIndex)

    return [
        firstAgent,
        secondAgent,
    ]


# --------------------------------------------------------------------------------------------------

"""
MIN MAX AGENT
"""
class MinMaxAgent(CaptureAgent):
    """
    A Dummy agent to serve as an example of the necessary agent structure.
    You should look at `pacai.core.baselineTeam` for more details about how to create an agent.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)
        self.treeDepth = 0

    def registerInitialState(self, gameState):
        """
        This method handles the initial setup of the agent and populates useful fields,
        such as the team the agent is on and the `pacai.core.distanceCalculator.Distancer`.

        IMPORTANT: If this method runs for more than 15 seconds, your agent will time out.
        """

        super().registerInitialState(gameState)
        self.treeDepth = 1  # our tree depth is 1 b/c it gets laggy when >1

    """
    CaptureAgent calls MinMaxAgent at each step.
    """
    def chooseAction(self, gameState):
        a = self.value(gameState, 0)

        return a

    """
    Call Function to MinMaxAgent Agent. Gets the best action per the algorithm.
    """
    def value(self, gameState, currentDepth):
        (v, a) = self.maxValue(gameState, currentDepth)
        return a

    """
    The evaluation function for the MinMaxAgent that determines which actions are preferable.
    What techniques we used in our evaluation function:
        - Track the number of enemy food, look to reduce it.
        - Split the two offensive agents to top / bot sides of the map.
        - Avoid enemy ghosts when in their territory.
        - Track the min distance to enemy food.
        - Function that helps reduce thrashing between two actions by discouraging turning around.
    """
    def evalFunction(self, gameState):
        # game state information
        # food
        enemyFood = self.getFood(gameState).asList()
        # agents
        ourAgents = self.getTeam(gameState)
        currentAgent = gameState.getAgentState(self.index)
        currentAgentPos = gameState.getAgentPosition(self.index)
        enemies = self.getOpponents(gameState)
        enemyAndPos = [(gameState.getAgentState(e), gameState.getAgentPosition(e)) for e in enemies]

        # track the number of food pellets left on the board
        numFood = 0
        if len(enemyFood) > 0:   # edge case for 0 pellets
            numFood = 1000 * (1 / len(enemyFood))

        # top / bot agent
        topBool = False                     # lower agent takes bot
        if self.index == max(ourAgents):
            topBool = True                  # higher agent takes top

        # avoid enemy ghosts in their territory
        avoidEnemyGhost = 0
        if currentAgent.isPacman():     # we are pacman
            for (e, p) in enemyAndPos:
                if e.isBraveGhost():    # enemy ghost is brave
                    d = self.getMazeDistance(currentAgentPos, p)
                    if d < 3:
                        avoidEnemyGhost = -6

        # track the distance to the closest food pellet on the board
        pacmanToFoodDistances = []
        for f in enemyFood:
            (xf, yf) = f

            # split the board in half
            if (topBool is True and yf >= 9) or (topBool is False and yf < 9):     # consider normal
                d = self.getMazeDistance(currentAgentPos, f)
                pacmanToFoodDistances.append(d)

        minfoodDist = 0
        # if only two pellets on one agent's side, also consider other side to avoid hyper-fixation.
        if len(pacmanToFoodDistances) > 2:      # think about only your side
            minfoodDist = min(pacmanToFoodDistances)
        else:                                   # also think about other side
            for f in enemyFood:
                d = self.getMazeDistance(currentAgentPos, f)
                pacmanToFoodDistances.append(d)
            minfoodDist = min(pacmanToFoodDistances)

        # anti-thrash algorithm
        prevState = self.getPreviousObservation()
        currState = self.getCurrentObservation()
        antiThrash = 0
        if prevState is not None and currState is not None and self.index == 2:
            if currentAgentPos == prevState.getAgentPosition(self.index):
                antiThrash = -5

        return numFood + avoidEnemyGhost - minfoodDist / 4.5 + antiThrash

    """
    Helper function to determine the depth of MinMaxAgent Tree
    """
    def getTreeDepth(self):
        return self.treeDepth

    """
    MaxValue part of MinMaxAgent. This is concerned with the actions of our agents.
    """
    def maxValue(self, gameState, currentDepth, alpha = -999999, beta = 999999):
        # terminal (including win/lose states)
        if currentDepth == self.getTreeDepth() or gameState.isWin() or gameState.isLose():
            return (self.evalFunction(gameState), None)
        # node
        else:
            # Get legal moves for this agent
            legalMoves = gameState.getLegalActions(self.index)
            if 'Stop' in legalMoves:    # remove 'Stop' move so that pacman always moving somwhere
                legalMoves.remove('Stop')

            # Get successor states from legal moves
            successors = [gameState.generateSuccessor(self.index, action) for action in legalMoves]

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

    """
    minValue part of MinMaxAgent. This is concerned with the actions of enemy ghost agents.
    Function will call itself for each enemy ghost (multiple min layers per max layer).
    """
    def minValue(self, gameState, currentDepth, agentNum = 0, alpha = -999999, beta = 999999):
        # terminal (including win/lose states)
        if currentDepth == self.getTreeDepth() or gameState.isWin() or gameState.isLose():
            return (self.evalFunction(gameState), None)
        # node
        else:
            # Get legal moves of ghost agent
            opponents = self.getOpponents(gameState)    # ghost agents
            legalMoves = gameState.getLegalActions(opponents[agentNum])
            if 'Stop' in legalMoves:    # remove 'Stop' move so that pacman always moving somwhere
                legalMoves.remove('Stop')

            # Get successor states from legal moves
            successors = [gameState.generateSuccessor(opponents[agentNum], action)
                          for action in legalMoves]

            # Send successors to minValue or maxValue func (multiple minValue layers, 1 per ghost)
            minPairs = []
            if agentNum == 1:    # all ghosts done
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
