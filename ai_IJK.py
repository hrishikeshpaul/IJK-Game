#!/usr/local/bin/python3

"""
This is where you should write your AI code!

Authors: Ishita Kumar (ishkumar), Hrishikesh Paul (hrpaul), Rushabh Shah (shah12)

Based on skeleton code by Abhilash Kuhikar, October 2019
"""
import math
from logic_IJK import Game_IJK
import random
import copy
import time
# Suggests next move to be played by the current player given the current game
#
# inputs:
#     game : Current state of the game 
#
# This function should analyze the current state of the game and determine the 
# best move for the current player. It should then call "yield" on that move.

PINF = 100000
NINF = -100000


"""
A class that is used to form the Minimax tree.
"""

class Node:
    def __init__(self, game, player, best_move, value=0):
        """
        :param game: Stores the configuration of the board
        :param player: Stores the current player for that board
        :param best_move: Stores the move for the board
        :param value: Stores the evaluation function value for the current board
        """

        self.game = game
        self.value = value
        self.player = player
        self.best_move = best_move



def next_move(game: Game_IJK)-> None:

    '''board: list of list of strings -> current state of the game
       current_player: int -> player who will make the next move either ('+') or -'-')
       deterministic: bool -> either True or False, indicating whether the game is deterministic or not
    '''

    board = game.getGame()
    player = game.getCurrentPlayer()
    deterministic = game.getDeterministic()

    # Create an object of the class Node
    node = Node(copy.deepcopy(board), player, '')
    start = time.time()
    if deterministic:
        # if deterministic, then call minimax with alpha beta pruning
        result = minimax_alpha_beta(node, 9, player, NINF, PINF, deterministic, start)
    else:
        # if non-deterministic then call normal minimax as expectation is involved
        result = minimax(node, 6, player, deterministic, start)
    print(time.time() - start)
    print(result)
    return result[0]
    # yield random.choice(['U', 'D', 'L', 'R'])


""""
The minimax algorithm starts here.
Code is inspired from the follow links:
http://blog.datumbox.com/using-artificial-intelligence-to-solve-the-2048-game-java-code/
https://github.com/speix/2048-puzzle-solver/blob/master/PlayerAI.py

"""


def minimax(node, depth, player, deterministic, start):

    """
    :param node: Object of class Node that contains the game, player, move and value
    :param depth: Depth of the tree
    :param player: Player of that node
    :param deterministic: Deterministic will have the value False as its minimax
    :param start: Start time of the algorithm
    :return: Returns the best move stored in the node and the value of the
    evaluation function
    """

    # If maximum depth is reached, or game is full, or the game state is the win
    # the time for each move exceeds the start time, then end the recursion by
    # returning the best move and the evaluation value

    if depth == 0 or isGameFull(node.game) or __game_state(node.game) != 0 or (time.time() - start) > 3:
        return (node.best_move, __evaluate(copy.deepcopy(node.game), player, deterministic))

    # Maximizing player
    if player == '+':
        best_value = NINF

        # Get an array of all possible moves from the current game
        possible_moves = generate_successors(node.game, player, deterministic)

        # Return the best move and the value if there are no possible moves
        if len(possible_moves) == 0:
            return (node.best_move, __evaluate(copy.deepcopy(node.game), player, deterministic))

        for child in possible_moves:
            # Recursively call minimax and get the value of the evaluation value
            value = minimax(copy.deepcopy(child), depth - 1, '-', deterministic, start)[1]

            # Update the best value based on the value from the recursion
            if value > best_value:
                best_value = value
                # Update the best move of the node
                node.best_move = child.best_move

        return (node.best_move, best_value)

    # Minimizing player
    else:
        best_value = PINF

        # Get an array of all possible moves from the current game
        possible_moves = generate_successors(node.game, player, deterministic)

        # Return the best move and the value if there are no possible moves
        if len(possible_moves) == 0:
            return (node.best_move, __evaluate(copy.deepcopy(node.game), player, deterministic))

        for child in possible_moves:
            # Recursively call minimax and get the value of the evaluation value
            value = minimax(copy.deepcopy(child), depth - 1, '+', deterministic, start)[1]

            # Update the best value based on the value from the recursion
            if value < best_value:
                best_value = value
                # Update the best move of the node
                node.best_move = child.best_move

        return (node.best_move, best_value)


"""
End of Mininmax Algorithm and inspired code.
"""

""""
The minimax algorithm with aplha beta pruning starts here
Code is inspired from the follow links:
http://blog.datumbox.com/using-artificial-intelligence-to-solve-the-2048-game-java-code/
https://github.com/deerishi/Tic-Tac-Toe-Using-Alpha-Beta-Minimax-Search/blob/master/tictac.py
https://github.com/hg2412/2048-AI/blob/master/PlayerAI.py
"""


def minimax_alpha_beta(node, depth, player, alpha, beta, deterministic, start):
    """

    :param node: Object of class Node that contains the game, player, move and value
    :param depth: Depth of the tree
    :param player: Player of that node
    :param deterministic: Deterministic will have the value True
    :param start: Start time of the algorithm
    :param alpha: Alpha value for pruning
    :param beta: Beta value for pruning
    :return: Returns the best move stored in the node and the value of the
    evaluation function
    """

    # If maximum depth is reached, or game is full, or the game state is the win
    # the time for each move exceeds the start time, then end the recursion by
    # returning the best move and the evaluation value

    if depth == 0 or isGameFull(node.game) or __game_state(node.game) != 0 or (time.time() - start) > 3:
        return (node.best_move, __evaluate(copy.deepcopy(node.game), player, deterministic))

    # Maximizing player
    if player == '+':
        best_value = NINF

        # Get an array of all possible moves from the current game
        possible_moves = generate_successors(node.game, player, deterministic)

        # Return the best move and the value if there are no possible moves
        if len(possible_moves) == 0:
            return (node.best_move, __evaluate(copy.deepcopy(node.game), player, deterministic))

        for child in possible_moves:
            # Recursively call minimax and get the value of the evaluation value
            value = minimax_alpha_beta(copy.deepcopy(child), depth - 1, '-', alpha, beta, deterministic, start)[1]
            # Update the best value based on the value from the recursion
            if value > best_value:
                best_value = value
                # Update the best move of the node
                node.best_move = child.best_move
            # Update the value of alpha
            if best_value > alpha:
                alpha = best_value
            # Prune the tree if alpha is greater than beta
            if alpha >= beta:
                break

        return (node.best_move, best_value)

    # Minimizing Player
    else:
        best_value = PINF
        # Get an array of all possible moves from the current game
        possible_moves = generate_successors(node.game, player, deterministic)

        # Return the best move and the value if there are no possible moves
        if len(possible_moves) == 0:
            return (node.best_move, __evaluate(copy.deepcopy(node.game), player, deterministic))

        for child in possible_moves:
            # Recursively call minimax and get the value of the evaluation value
            value = minimax_alpha_beta(copy.deepcopy(child), depth - 1, '+', alpha, beta, deterministic, start)[1]
            # Update the best value based on the value from the recursion
            if value < best_value:
                best_value = value
                # Update the best move of the node
                node.best_move = child.best_move
            # Update the value of beta
            if best_value < beta:
                beta = best_value
            # Prune the tree if alpha is greater than beta
            if alpha >= beta:
                break

        return (node.best_move, best_value)

"""
Minimax with Alpha Beta algorithm and inspired code ends here.
"""

"""
Calculates the number of empty blocks for a game configuration
"""


def __empty_blocks(mat):
    """
    :param mat: Game configuration
    :return: Number of empty blocks
    """
    if not isGameFull(mat):
        empty_blocks = 0
        for i in range(len(mat)):
            for j in range(len(mat)):
                if mat[i][j] == ' ':
                    empty_blocks += 1
        return empty_blocks

    else:
        return 0

"""
Calculates the maximum number of merges possible of a given configuration for
a player.
"""


def __max_merge_heuristic(mat, player):
    """
    :param mat: Game configuration
    :param player: '+' or '-'
    :return: Total number of merges possible
    """

    total_merges = 0

    for i in range(len(mat)):
        for j in range(len(mat) - 1):
            if mat[i][j] == mat[i][j + 1] and mat[i][j] != ' ':
                mat[i][j] = chr(ord(mat[i][j]) + 1)
                mat[i][j + 1] = ' '
                total_merges = total_merges + 1
            elif mat[i][j].upper() == mat[i][j + 1].upper() and mat[i][
                j] != ' ':
                mat[i][j] = chr(ord(mat[i][j]) + 1)
                mat[i][j] = mat[i][j].upper() if player == "+" else \
                mat[i][j].lower()
                mat[i][j + 1] = ' '
                total_merges = total_merges + 1
    return total_merges


"""
Calculates the smoothness value for a given game. For each block, the calculate 
if it is within 1 manhattan distance away from the tile with similar numbers.
Explained in detail in the documentation.

Code inspired from https://github.com/speix/2048-puzzle-solver/blob/master/PlayerAI.py
"""


def __smoothness_heuristic(mat):
    """
    :param mat: game configuration
    :return: Smoothness value
    """
    smoothness = 0

    for x in range(len(mat)):
        for y in range(len(mat)):
            s = float('infinity')

            if x > 0:
                s = min(s,
                        abs((ord(mat[x][y].upper())) - (ord(mat[x-1][y].upper()))))
            if y > 0:
                s = min(s,
                        abs((ord(mat[x][y].upper())) - (ord(mat[x][y-1].upper()))))
            if x < 5:
                s = min(s,
                        abs((ord(mat[x][y].upper())) - (ord(mat[x+1][y].upper()))))
            if y < 5:
                s = min(s,
                        abs((ord(mat[x][y].upper())) - (ord(mat[x][y+1].upper()))))

            smoothness -= s

    return smoothness

"""
Inspired code ends here 
"""

"""
Evaluation function. If the mode is non-deterministic, it multiplies the value 
by the chance, which is 1/n (n = number of empty blocks)
"""


def __evaluate(mat, player, deterministic):
    """
    :param mat: Game configuration
    :param player: Player of the game
    :param deterministic: Determinism value to multiply the chance of a node
    :return: Value of the evalutation
    """

    if deterministic:
        # Sum of evaluations multiplied by some weights
        return __max_merge_heuristic(mat, player) * 1.2 + __smoothness_heuristic(mat) * 0.1 + __empty_blocks(mat) * 2.7
    else:
        # Sum of evaluations multiplied by some weights
        h_res = __max_merge_heuristic(mat, player) * 1.2 + __smoothness_heuristic(mat) * 0.1 + __empty_blocks(mat) * 2.7
        # Result multiplied by the chance
        return h_res * 1/__empty_blocks(mat) if not isGameFull(mat) else 0


"""
Generates successors for the given game configuration
"""


def generate_successors(board, player, deterministic):
    """

    :param board: Board configuration
    :param player: Player of the node
    :param deterministic: Determinism value
    :return:
    """
    available = []
    for move in ['U', 'D', 'L', 'R']:
        # Call makeMove to get the new board configuration
        (new_board, _) = makeMove(copy.deepcopy(board), move, player, deterministic)

        if new_board:
            new_node = Node(new_board, player, move)
            available.append(new_node)

    return available

"""
Functions below have been taken from logic_IJK.py
"""


def makeMove(board, move, player, deterministic):
    doneBoard = []
    done = False

    if move == 'L':
        game, done = __cover_up(board)
        temp = __merge(game, player)
        game = temp[0]
        done = done or temp[1]
        game = __cover_up(game)[0]
        if done == True:
            doneBoard = copy.deepcopy(game)

    if move == 'U':
        game = __transpose(board)
        game, done = __cover_up(game)
        temp = __merge(game, player)
        game = temp[0]
        done = done or temp[1]
        game = __cover_up(game)[0]
        game = __transpose(game)
        if done == True:
            doneBoard = copy.deepcopy(game)

    if move == 'D':
        game = __reverse(__transpose(copy.deepcopy(board)))
        game, done = __cover_up(game)
        temp = __merge(board, player)
        game = temp[0]
        done = done or temp[1]
        game = __cover_up(game)[0]
        game = __transpose(__reverse(game))
        if done == True:
            doneBoard = copy.deepcopy(game)

    if move == 'R':
        game = __reverse(board)
        game, done = __cover_up(game)
        temp = __merge(game, player)
        game = temp[0]
        done = done or temp[1]
        game = __cover_up(game)[0]
        game = __reverse(game)
        if done == True:
            doneBoard = copy.deepcopy(game)

    __add_piece(doneBoard, player, deterministic)
    return (doneBoard, done)


def __add_piece(board,player, deterministic):
    if deterministic:
        for i in range(len(board)):
            for j in range(len(board)):
                if board[i][j] == ' ':
                    board[i][
                        j] = 'A' if player == '+' else 'a'
                    return
    else:
        open = []
        for i in range(len(board)):
            for j in range(len(board)):
                if board[i][j] == ' ':
                    open += [(i, j), ]

        if len(open) > 0:
            r = random.choice(open)
            board[r[0]][r[1]] = 'A' if player == '+' else 'a'
            return


def __cover_up(mat):
    new = [[' ' for _ in range(len(mat))] for _ in
           range(len(mat))]

    done = False
    for i in range(len(mat)):
        count = 0
        for j in range(len(mat)):
            if mat[i][j] != ' ':
                new[i][count] = mat[i][j]
                if j != count:
                    done = True
                count += 1
    return (new, done)


def __merge(mat, player):

    done = False
    for i in range(len(mat)):
        for j in range(len(mat) - 1):
            if mat[i][j] == mat[i][j + 1] and mat[i][j] != ' ':
                mat[i][j] = chr(ord(mat[i][j]) + 1)
                mat[i][j + 1] = ' '
                done = True
            elif mat[i][j].upper() == mat[i][j + 1].upper() and mat[i][
                j] != ' ':
                mat[i][j] = chr(ord(mat[i][j]) + 1)
                mat[i][j] = mat[i][
                    j].upper() if player == '+' else mat[i][
                    j].lower()
                mat[i][j + 1] = ' '
                done = True
    return (mat, done)


def __transpose(mat):
    new = []
    for i in range(len(mat[0])):
        new.append([])
        for j in range(len(mat)):
            new[i].append(mat[j][i])
    return new


def __reverse(mat):
    new = []
    for i in range(len(mat)):
        new.append([])
        for j in range(len(mat[0])):
            new[i].append(mat[i][len(mat[0]) - j - 1])
    return new


def isGameFull(board):
    for i in range(len(board)):
        for j in range(len(board)):
            if board[i][j] == ' ':
                return False
    return True


def __game_state(mat):
    highest = {'+': 'A', '-': 'a'}

    for i in range(len(mat)):
        for j in range(len(mat[0])):
            if (mat[i][j]).isupper():
                highest['+'] = chr(max(ord(mat[i][j]), ord(highest['+'])))
            if (mat[i][j]).islower():
                highest['-'] = chr(max(ord(mat[i][j]), ord(highest['-'])))

    if highest['+'] == 'K' or highest['-'] == 'k' or isGameFull(mat):
        if highest['+'].lower() != highest['-']:
            return highest['+'] if highest['+'].lower() > highest['-'] else \
                highest['-']
        return 'Tie'

    return 0
