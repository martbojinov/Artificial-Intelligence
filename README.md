# Artificial-Intelligence
Projects done during my time learning artificial intelligence.

Projects were done in a modified version of the Pacman educational project from the [Berkeley AI Lab](http://ai.berkeley.edu/project_overview.html).

Projects consist of:

1) Search - Implementing various search algorithms to reach the end of a maze. Includes BFS, DFS, UniformCostSearch, A* Search and applying these search methods to various methods / problems. Each problem is defined by a unique heuristic in which the agent is utilizing the search algorithms to select specific goals as defined by the hueristic.

2) Games / Adversaries / Multiagents (Reflex, Minimax, and Expectimax) - This assumes that there is an adversarial agent in the game. We are focused on creating agents that either react on current state or by looking into the future and seeing possible states. Reflex agent reacts on current state without any thought for the future. Minimax agent uses an evaluation function to give each possible future game state a ranking. It assumes that the adversary will take the action that will minimize rewards so it will look to take the action that will maximize in return to that. This is evaluated in a tree of depth n where the depth is equal to how many turns in the future we want to look. Increasing depth n severly incereases computation time and can be very costly. Alpha-Beta pruning is implemented to cut costly branches which are irrelavent, increasing efficiency. Finally expectimax is used where the adversary instead of minimizing takes a random action. In that case we take the average of all the the adversaries actions to try and act accordingly.

3) Markov Decision Processes (MDPS) and Reinforcement Learning (Q-learning and Value Iteration) - In the previous problems, we assumed that we had total knowledge of our environments. In reinforcement learning, we know nothing and have to build a policy based on actions we take and their varying success / failure. We use Q-learning and Value Iteration as techniques to build our policy. After x amount of iterations the policy converges to the optimal policy and we can utilize the agent.

4) Pacman Capture the Flag Contest (1st Place) - TBD
