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


import random

import util
import time
from game import Agent, Directions  # noqa
from util import manhattanDistance  # noqa

inf = 10000000


class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        newSuccessorGameState = currentGameState.generatePacmanSuccessor(action)
        newGhostStates = newSuccessorGameState.getGhostStates()
        newPos = newSuccessorGameState.getPacmanPosition()

        if newSuccessorGameState.isWin():
          return inf
        
        newGhostPositions = []
        for ghostState in newGhostStates:
          newGhostPositions.append(ghostState.getPosition())
        
        ghostDistances = list(map(lambda x: manhattanDistance(x, newPos), newGhostPositions))
        if min(ghostDistances) == 2:
          return -inf/3
        elif min(ghostDistances) == 1:
          return -2*inf/3
        elif min(ghostDistances) == 0:
          return -inf

        ghostDistanceSum = sum(ghostDistances)


        newFoodList = newSuccessorGameState.getFood().asList()
        manhattanFoodList = map(lambda x: util.manhattanDistance(newPos, x), newFoodList)
        distanceToClosestFood = min(manhattanFoodList)
        newFoodLeft = len(newFoodList)
        if newFoodLeft < len(currentGameState.getFood().asList()):
          distanceToClosestFood = 0

        currentScore = scoreEvaluationFunction(newSuccessorGameState)
        score =  1 * currentScore\
                -2 * distanceToClosestFood\
                -1 * newFoodLeft\
                -1 * (1./ghostDistanceSum)

        return score


def scoreEvaluationFunction(currentGameState):
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

    def __init__(self, evalFn="scoreEvaluationFunction", depth="2"):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    def getAction(self, gameState):
        def minimax(state, depth, agentIndex):
          if state.isWin() or state.isLose():
            return state.getScore()
          if agentIndex == 0: # PAC-MAN node (max node)
            newAgentIndex = agentIndex + 1
            maxScore = -inf
            maxAction = Directions.STOP
            # need to generate successors
            actions = state.getLegalActions(agentIndex)
            # need to generate scores for each successor:
            for action in actions:
              successorState = state.generateSuccessor(agentIndex, action)
              score = minimax(successorState, depth, newAgentIndex)
              if score > maxScore:
                maxScore = score
                maxAction = action
            if depth == 0:
              return maxAction
            else:
              return maxScore
          else:
            minScore = inf
            # need to generate successors
            actions = state.getLegalActions(agentIndex)
            # if we are looking at the final ghost, increment the depth:
            if agentIndex == state.getNumAgents() - 1:
              depth += 1
              newAgentIndex = 0
            else:
              newAgentIndex = agentIndex + 1
            # need to generate scores for each successor:
            for action in actions:
              # if we are at depth, evaluate the state
              if depth == self.depth:
                score = self.evaluationFunction(state.generateSuccessor(agentIndex, action))
              # otherwise, go deeper
              else:
                successorState = state.generateSuccessor(agentIndex, action)
                score = minimax(successorState, depth, newAgentIndex)
              minScore = min(score, minScore)
            return minScore
        return minimax(gameState, 0, 0)

  
class AlphaBetaAgent(MultiAgentSearchAgent):
    def getAction(self, gameState):
        def alphaBetaMinimax(state, depth, agentIndex, alpha, beta):
          if state.isWin() or state.isLose():
            return state.getScore()
          if agentIndex == 0: # PAC-MAN node (max node)
            newAgentIndex = agentIndex + 1
            maxAction = Directions.STOP
            # need to generate successors
            actions = state.getLegalActions(agentIndex)
            # need to generate scores for each successor:
            for action in actions:
              successorState = state.generateSuccessor(agentIndex, action)
              score = alphaBetaMinimax(successorState, depth, newAgentIndex, alpha, beta)
              if score > alpha:
                alpha = score
                maxAction = action
              if alpha >= beta:
                break
            if depth == 0:
              return maxAction
            else:
              return alpha
          else:
            # need to generate successors
            actions = state.getLegalActions(agentIndex)
            # if we are looking at the final ghost, increment the depth:
            if agentIndex == state.getNumAgents() - 1:
              depth += 1
              newAgentIndex = 0
            else:
              newAgentIndex = agentIndex + 1
            # need to generate scores for each successor:
            for action in actions:
              # if we are at depth, evaluate the state
              if depth == self.depth:
                score = self.evaluationFunction(state.generateSuccessor(agentIndex, action))
              # otherwise, go deeper
              else:
                successorState = state.generateSuccessor(agentIndex, action)
                score = alphaBetaMinimax(successorState, depth, newAgentIndex, alpha, beta)
              beta = min(score, beta)
              if alpha >= beta:
                break
            return beta
        return alphaBetaMinimax(gameState, 0, 0, -inf, inf)


class ExpectimaxAgent(MultiAgentSearchAgent):
    def getAction(self, gameState):
        def expectimax(state, depth, agentIndex):
          if state.isWin() or state.isLose():
            return state.getScore()
          if agentIndex == 0: # PAC-MAN node (max node)
            newAgentIndex = agentIndex + 1
            maxScore = -inf
            maxAction = Directions.STOP
            # need to generate successors
            actions = state.getLegalActions(agentIndex)
            # need to generate scores for each successor:
            for action in actions:
              successorState = state.generateSuccessor(agentIndex, action)
              score = expectimax(successorState, depth, newAgentIndex)
              if score > maxScore:
                maxScore = score
                maxAction = action
            if depth == 0:
              return maxAction
            else:
              return maxScore
          else:
            # need to generate successors
            actions = state.getLegalActions(agentIndex)
            # if we are looking at the final ghost, increment the depth:
            if agentIndex == state.getNumAgents() - 1:
              depth += 1
              newAgentIndex = 0
            else:
              newAgentIndex = agentIndex + 1
            # need to generate scores for each successor:
            totalScore = 0
            for action in actions:
              # if we are at depth, evaluate the state
              if depth == self.depth:
                totalScore += self.evaluationFunction(state.generateSuccessor(agentIndex, action))
              # otherwise, go deeper
              else:
                successorState = state.generateSuccessor(agentIndex, action)
                totalScore += expectimax(successorState, depth, newAgentIndex)
            averageScore = totalScore/len(actions)
            return averageScore
        return expectimax(gameState, 0, 0)


def betterEvaluationFunction(currentGameState):
    ghostStates = currentGameState.getGhostStates()
    pos = currentGameState.getPacmanPosition()
    foodList = currentGameState.getFood().asList()
    capsuleList = list(currentGameState.getCapsules())

    if currentGameState.isWin():
      return inf
    if currentGameState.isLose():
      return -inf


    normalGhostPositions = []
    scaredGhostPositions = []
    for ghostState in ghostStates:
      if ghostState.scaredTimer > 15:
        scaredGhostPositions.append(ghostState.getPosition())
      else:
        normalGhostPositions.append(ghostState.getPosition())

    normalGhostDistances = list(map(lambda x: manhattanDistance(x, pos), normalGhostPositions))
    scaredGhostDistances = list(map(lambda x: manhattanDistance(x, pos), scaredGhostPositions))
    if len(scaredGhostDistances) > 0:
      nearestScaredGhost = min(scaredGhostDistances)
    else:
      nearestScaredGhost = 1

    if len(normalGhostDistances) > 0:  
      if min(normalGhostDistances) == 1:
        return -inf/2
      elif min(normalGhostDistances) == 0:
        return -inf
      ghostDistanceSum = sum(normalGhostDistances)
    else:
      ghostDistanceSum = 1

    numNormalGhosts = len(normalGhostPositions)

    manhattanCapsuleList = list(map(lambda x: util.manhattanDistance(x, pos), capsuleList))
    if len(manhattanCapsuleList) > 0:
      distanceToNearestCapsule = min(manhattanCapsuleList)
    else:
      distanceToNearestCapsule = 0

    manhattanFoodList = list(map(lambda x: util.manhattanDistance(x, pos), foodList))
    distanceToNearestFood = min(manhattanFoodList)

    foodLeft = len(foodList)

    currentScore = scoreEvaluationFunction(currentGameState)
    score =  1 * currentScore\
            -1 * nearestScaredGhost\
            -1 * distanceToNearestCapsule\
            -1 * distanceToNearestFood\
            -1 * foodLeft\
            -100 * numNormalGhosts\
            -1 * (1./ghostDistanceSum)

    return score


# Abbreviation
better = betterEvaluationFunction
