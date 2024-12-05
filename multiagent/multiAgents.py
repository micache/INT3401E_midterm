# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        score = successorGameState.getScore()
        # lấy vị trí các food
        foodList = newFood.asList()
        # Cân nhắc khoảng cách đến food, ưu tiên food gần nhất
        if foodList: 
            minFoodDistance = min([manhattanDistance(newPos, foodPosition) for foodPosition in foodList])
            score += 1.0/minFoodDistance
        # Cân nhắc khoảng cách với ma
        for ghost, scaredTime in zip(newGhostStates, newScaredTimes):
            # tính khoảng cách đến các ma
            ghostDistance = manhattanDistance(newPos, ghost.configuration.pos) 
            # nếu ma sợ đến ăn ma
            if scaredTime > 0:
                score += 10.0/ghostDistance if ghostDistance > 0 else 200
            # nếu ma không sợ và cách ma quá gần, tránh xa ma
            else:
                if ghostDistance < 2:
                    score -= 10
        # phạt nếu pacman đứng yên
        if action == 'Stop':
            score -= 50
        
        return score

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        result = self.minimax(gameState, 0, self.depth)

        return result[1]
    
    #Lấy ra giá trị Max của Max-agent
    def max_value(self, gameState: GameState, index, depth):
        action_list = []
        for action in gameState.getLegalActions(index):
            action_list.append((self.minimax(gameState.generateSuccessor(index, action), index + 1, depth)[0], action))
        return max(action_list)
    
    #Lấy ra giá trị Min của Min-agent
    def min_value(self, gameState: GameState, index, depth):
        action_list = []
        for action in gameState.getLegalActions(index):
            action_list.append((self.minimax(gameState.generateSuccessor(index, action), index + 1, depth)[0], action))
        return min(action_list)
    
    #Tiến hành thuật toán tìm kiếm
    def minimax(self, gameState: GameState, index, depth):
        #Trả về kết quả cuối cùng
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return (self.evaluationFunction(gameState), "")
        
        #Kiểm tra và cập nhật độ sâu của quá trình tìm kiếm
        agentsNum = gameState.getNumAgents()
        index %= agentsNum
        if index == agentsNum - 1 :
            depth -= 1
        
        if index == 0:
            #Max-agent: Pacman
            return self.max_value(gameState, index, depth)
        else:
            #Min-agent: Ma
            return self.min_value(gameState, index, depth)

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.minimax(gameState, 0, self.depth)[1]

    #Lấy ra giá trị Max cho Max-agent
    def maxValue(self, gameState: GameState, index, depth, alpha, beta):
        action_list = []
        for action in gameState.getLegalActions(index):
            v = self.minimax(gameState.generateSuccessor(index, action), index + 1, depth, alpha, beta)[0]
            action_list.append((v, action))
            if v > beta:
                return (v, action)
            alpha = max(alpha, v)
        return max(action_list)       
    
    #Lấy ra giá trị Min cho Min-agent
    def minValue(self, gameState: GameState, index, depth, alpha, beta):
        action_list = []
        for action in gameState.getLegalActions(index):
            v = self.minimax(gameState.generateSuccessor(index, action), index + 1, depth, alpha, beta)[0]
            action_list.append((v, action))
            if v < alpha:
                return (v, action)
            beta = min(beta, v)
        return min(action_list)
    
    def minimax(self, gameState: GameState, index, depth, alpha = -999999, beta = 999999):
        #Trả về kết quả cuối cùng
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return ( self.evaluationFunction(gameState), "")
        
        #Kiểm tra và cập nhật độ sâu tìm kiếm
        agentsNum = gameState.getNumAgents()
        index %= agentsNum
        if index == agentsNum - 1:
            depth -= 1

        if index == 0:
            #Max-agent: Pacman
            return self.maxValue(gameState, index, depth, alpha, beta)
        else:
            #Min-agent: Ma
            return self.minValue(gameState, index, depth, alpha, beta)

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"

        res, score = self.cal_point(0, 0, gameState)
        return res
        util.raiseNotDefined()

    def cal_point(self, depth: int, id: int, gamestate: GameState):

        # Terminal state
        if len(gamestate.getLegalActions(id)) == 0 or depth == self.depth:
            return "", self.evaluationFunction(gamestate)
        
        # check if pacman or not
        if id == 0:
            return self.find_max(depth, id, gamestate)
        else:
            return self.expectimax(depth, id, gamestate)
        
    def find_max(self, depth: int, id: int, gamestate: GameState):
        
        max_point = float("-inf")
        max_action = ""

        for move in gamestate.getLegalActions(id):
            state = gamestate.generateSuccessor(id, move)
            state_id = (id + 1) % gamestate.getNumAgents()
            state_depth = depth

            # is pacman, so go down a step
            if state_id == 0:
                state_depth += 1

            _, state_point = self.cal_point(state_depth, state_id, state)

            if max_point < state_point:
                max_point = state_point
                max_action = move
        
        return max_action, max_point
    
    def expectimax(self, depth: int, id: int, gamestate: GameState):
        
        expect_point = 0.
        prob = 1.0 / len(gamestate.getLegalActions(id))

        for move in gamestate.getLegalActions(id):

            state = gamestate.generateSuccessor(id, move)
            state_id = (id + 1) % gamestate.getNumAgents()
            state_depth = depth

            # is pacman, so go down a step
            if state_id == 0:
                state_depth += 1

            state_action, state_point = self.cal_point(state_depth, state_id, state)
            expect_point += prob * state_point
        
        return "", expect_point


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
   # Lấy thông tin cơ bản từ trạng thái hiện tại
    pacmanPos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghost.scaredTimer for ghost in ghostStates]
    score = currentGameState.getScore()
    foodList = food.asList()

    if foodList:
        closestFood = min(manhattanDistance(pacmanPos, foodPos) for foodPos in foodList)
        score += 10 / closestFood  # Ưu tiên ăn thức ăn gần nhất

    # 2. Cân nhắc khoảng cách với ma
    for ghost, scaredTime in zip(ghostStates, scaredTimes):
        ghostDistance = manhattanDistance(pacmanPos, ghost.getPosition())

        if scaredTime > 0:  # Ma đang sợ, Pacman có thể săn ma
            score += 200 / ghostDistance if ghostDistance > 0 else 500
        else:  # Ma không sợ, tránh xa ma
            if ghostDistance < 2:
                score -= 500  # Phạt nặng nếu quá gần ma

    # 3. Phạt nặng nếu còn quá nhiều thức ăn
    score -= len(foodList) * 10

    # 4. Phạt nếu Pacman đứng yên
    if currentGameState.getPacmanPosition() == pacmanPos:
        score -= 10

    return score
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction