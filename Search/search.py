"""
In this file, you will implement generic search algorithms which are called by Pacman agents.
"""

from pacai.util import stack
from pacai.util import queue
from pacai.util import priorityQueue

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first [p 85].

    Your search algorithm needs to return a list of actions that reaches the goal.
    Make sure to implement a graph search algorithm [Fig. 3.7].

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
    ```
    print("Start: %s" % (str(problem.startingState())))
    print("Is the start a goal?: %s" % (problem.isGoal(problem.startingState())))
    print("Start's successors: %s" % (problem.successorStates(problem.startingState())))
    ```
    """

    nodeDict = {}                                   # dictionary storing child : (parent,  action)
    fringe = stack.Stack()                          # create fringe (stack)
    fringe.push(problem.startingState())
    visitHistory = []                               # history of visited nodes

    while not fringe.isEmpty():
        node = fringe.pop()
        if problem.isGoal(node):                    # reached goal
            (parent, action, cost, aList) = nodeDict[node]
            actionList = list(aList)
            return actionList
        if node not in visitHistory:                # expand successors to stack (avoiding explored)
            for (state, action, cost) in problem.successorStates(node):
                if state not in visitHistory:
                    # caculate forward cost and actionlist to successor
                    if node == problem.startingState():     # node is root
                        aList = []
                        aList.append(action)
                        nodeDict[state] = (node, action, cost, tuple(aList))
                    else:                                   # node is non root
                        (pState, pAction, pCost, pList) = nodeDict[node]    # Parent node
                        pList = list(pList)
                        pList.append(action)
                        nodeDict[state] = (node, action, cost + pCost, tuple(pList))
                fringe.push(state)
        visitHistory.append(node)

    raise "No DFS path avalaible"           # stack is emptied, failed to find goal

def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first. [p 81]
    """

    nodeDict = {}                                   # dictionary storing child : (parent,  action)
    fringe = queue.Queue()                          # create fringe (queue)
    fringe.push(problem.startingState())
    visitHistory = []                               # history of visited nodes

    while not fringe.isEmpty():
        node = fringe.pop()
        if problem.isGoal(node):                    # reached goal
            (parent, action, cost, aList) = nodeDict[node]
            actionList = list(aList)
            return actionList
        if node not in visitHistory:                # expand successors to stack (avoiding explored)
            for (state, action, cost) in problem.successorStates(node):
                if state not in visitHistory:
                    # caculate forward cost and actionlist to successor
                    if node == problem.startingState():     # node is root
                        aList = []
                        aList.append(action)
                        nodeDict[state] = (node, action, cost, tuple(aList))
                    else:                                   # node is non root
                        (pState, pAction, pCost, pList) = nodeDict[node]    # Parent node
                        pList = list(pList)
                        pList.append(action)
                        nodeDict[state] = (node, action, cost + pCost, tuple(pList))
                fringe.push(state)
        visitHistory.append(node)

    raise "No BFS path avalaible"           # queue is emptied, failed to find goal

def uniformCostSearch(problem):
    """
    Search the node of least total cost first.
    """

    nodeDict = {}                                   # dictionary storing child : (parent,  action)
    fringe = priorityQueue.PriorityQueue()          # create fringe (priority queue)
    fringe.push(problem.startingState(), 0)
    visitHistory = []                               # history of visited nodes

    while not fringe.isEmpty():
        node = fringe.pop()
        if problem.isGoal(node):                    # reached goal
            (parent, action, cost, aList) = nodeDict[node]
            actionList = list(aList)
            return actionList
        if node not in visitHistory:                # expand successors to stack (avoiding explored)
            for (state, action, cost) in problem.successorStates(node):
                if state not in visitHistory:
                    # caculate forward cost and actionlist to successor
                    if node == problem.startingState():     # node is root
                        aList = []
                        aList.append(action)
                        nodeDict[state] = (node, action, cost, tuple(aList))
                    else:                                   # node is non root
                        (pState, pAction, pCost, pList) = nodeDict[node]    # Parent node
                        pList = list(pList)
                        pList.append(action)
                        nodeDict[state] = (node, action, cost + pCost, tuple(pList))
                fringe.push(state, cost)
        visitHistory.append(node)

    raise "No UCS path avalaible"           # priority queue is emptied, failed to find goal

def aStarSearch(problem, heuristic):
    """
    Search the node that has the lowest combined cost and heuristic first.
    """

    nodeDict = {}                                   # dictionary of child : (parent,  action, cost)
    fringe = priorityQueue.PriorityQueue()          # create fringe (priority queue)
    fringe.push(problem.startingState(), 0 + heuristic(problem.startingState(), problem))
    visitHistory = []                               # history of visited nodes

    while not fringe.isEmpty():
        node = fringe.pop()
        if problem.isGoal(node):                    # reached goal
            (parent, action, cost, aList) = nodeDict[node]
            actionList = list(aList)
            return actionList
        if node not in visitHistory:                # expand successors to stack (avoiding explored)
            for (state, action, cost) in problem.successorStates(node):
                totalCost = 1
                if state not in visitHistory:
                    # caculate forward cost and actionlist to successor
                    if node == problem.startingState():     # node is root
                        aList = []
                        aList.append(action)
                        nodeDict[state] = (node, action, cost, tuple(aList))
                        totalCost = cost
                    else:                                   # node is non root
                        (pState, pAction, pCost, pList) = nodeDict[node]    # Parent node
                        pList = list(pList)
                        pList.append(action)
                        nodeDict[state] = (node, action, cost + pCost, tuple(pList))
                        totalCost = cost + pCost
                fringe.push(state, totalCost + heuristic(state, problem))
        visitHistory.append(node)

    raise "No A* path avalaible"            # priority queue is emptied, failed to find goal
