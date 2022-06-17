# Grupo 30:
# 92759 Laura Quintas
# 92780 Raquel Romão

#import sys (como estava antes)


from sys import stdin
import copy
import numpy as np
from search import (
    Problem,
    Node,
    astar_search,
    breadth_first_tree_search,
    depth_first_tree_search,
    greedy_search,
    recursive_best_first_search,
)


class TakuzuState:
    state_id = 0

    def __init__(self, board):
        self.board = board
        self.id = TakuzuState.state_id
        TakuzuState.state_id += 1
        self.empty = board.empty
        self.open = False
        self.possible_actions = None
        #self.filled_rows = set()
        #self.filled_cols = set()

    def __lt__(self, other):
        return self.id < other.id

    def __hash__(self):
        return hash(self.board)


    def actions(self):
        if not self.open:
            if self.possible_actions == None:
                line = list(zip((self.board.board==0).sum(axis=1), (self.board.board==1).sum(axis=1)))
                col = list(zip((self.board.board==0).sum(axis=0), (self.board.board==1).sum(axis=0)))
                actions = []
                
                empty = self.board.empty


                if self.board.board_size % 2 == 0:
                    half = self.board.board_size //2
                else:
                    half = self.board.board_size //2 + 1


                for i in empty:
                    position_actions = []

                    if line[i[0]][0] < half and col[i[1]][0] < half and self.board.adjacent_vertical_numbers(i[0],i[1]).count(0)!=2 and self.board.adjacent_horizontal_numbers(i[0],i[1]).count(0)!=2:
                        
                        position_actions.append((i[0],i[1],0))

                    if line[i[0]][1] < half and col[i[1]][1] < half and self.board.adjacent_vertical_numbers(i[0],i[1]).count(1)!=2 and self.board.adjacent_horizontal_numbers(i[0],i[1]).count(1)!=2:
                        position_actions.append((i[0],i[1],1))
                    
                    if len(position_actions)==2:
                        actions.insert(0,position_actions[0])
                        actions.insert(0,position_actions[1])
                    
                    elif len(position_actions)==1:
                        actions.append(position_actions[0])

                    else:
                        self.possible_actions = []
                        return self.possible_actions

                self.possible_actions = actions
    
            return self.possible_actions
        else:
            self.possible_actions = []
            return self.possible_actions


    #def find_broken_rule(self, row: int, col: int, value: int):



    def expand(self):
        self.open = True


class Board:
    """Representação interna de um tabuleiro de Takuzu."""
    def __init__(self, board, board_size: int, empty: list):
        self.board = board
        self.board_size = board_size
        self.empty = empty
        self.string = str(self.board.ravel())

    def __repr__(self):
        prettyprint = ''
        for i in self.board:
            for j in range(len(i)):
                if j == len(i)-1:
                    prettyprint += f'{i[j]}\n'
                else:
                    prettyprint += f'{i[j]}    '
        return prettyprint


    def set_number(self, row: int, col: int, value): 
        self.board[row, col] = value
        self.empty.remove((row,col))
        self.string = str(self.board.ravel()) # atualiza o hash value -> não sei se é necessário atualizar aqui ou se atualiza depois automaticamente


    def get_number(self, row: int, col: int):
        """Devolve o valor na respetiva posição do tabuleiro."""
        return self.board[row, col]


    def adjacent_vertical_numbers(self, row: int, col: int):
        """Devolve os valores imediatamente abaixo e acima,
        respectivamente."""
        if row == 0:
            return (None, self.get_number(row + 1, col))
        
        elif row == self.board_size - 1:
            return (self.get_number(row - 1, col), None)

        else:
            return (self.get_number(row - 1, col), self.get_number(row + 1, col))


    def adjacent_horizontal_numbers(self, row: int, col: int):
        """Devolve os valores imediatamente à esquerda e à direita,
        respectivamente."""
        if col == 0:
            return (None, self.get_number(row, col + 1))
        
        elif col == self.board_size - 1:
            return (self.get_number(row, col - 1), None)

        else:
            return (self.get_number(row, col - 1), self.get_number(row, col + 1))

    
    def __hash__(self):
        return hash(self.string)


    def copy(self):
        new_board = self.board.copy()
        empty = copy.copy(self.empty)
        return Board(new_board, self.board_size, empty)


    @staticmethod
    def parse_instance_from_stdin():
        """Lê o test do standard input (stdin) que é passado como argumento
        e retorna uma instância da classe Board.

        Por exemplo:
            $ python3 takuzu.py < input_T01

            > from sys import stdin
            > stdin.readline()
        """
        board_size = int(stdin.readline().rstrip('\n'))
        board = np.ones((board_size, board_size), dtype=int)
        empty = []
        for i in range(board_size):
            values = stdin.readline().strip('\n').split('\t') 
            for j in range(board_size):
                value = int(values[j])
                board[i, j] = value
                if value == 2:
                    empty.append((i,j))


        new_board = Board(board, board_size, empty)

        return new_board

    




class Takuzu(Problem):
    def __init__(self, board: Board):
        """O construtor especifica o estado inicial."""
        self.initial = TakuzuState(board)
        self.visited_states = {}


    def actions(self, state: TakuzuState):
        """Retorna uma lista de ações que podem ser executadas a
        partir do estado passado como argumento."""
        actions = state.actions()
        state.expand()
        return actions


    def result(self, state: TakuzuState, action):
        """Retorna o estado resultante de executar a 'action' sobre
        'state' passado como argumento. A ação a executar deve ser uma
        das presentes na lista obtida pela execução de
        self.actions(state)."""
        new_board = state.board.copy()

        new_board.set_number(action[0], action[1], action[2])

        hash_state = hash(new_board)

        if hash_state in self.visited_states:

            return self.visited_states[hash_state]

        new_state = TakuzuState(new_board)
        self.visited_states.update({hash_state: new_state})
        
        return new_state


    def dif_rows_cols(self, state: TakuzuState):
        _, row_counts = np.unique(state.board.board, axis=0, return_counts=True)
        unique_rows = len(row_counts) == state.board.board_size

        _, col_counts = np.unique(state.board.board, axis=1, return_counts=True)
        unique_cols = len(col_counts) == state.board.board_size

        return unique_rows and unique_cols


    def half_half(self, state: TakuzuState):
        board_size = state.board.board_size
        half = board_size // 2
        sum_col = np.sum(state.board.board, axis=0)
        sum_lines = np.sum(state.board.board, axis=1)
    
        if board_size % 2 == 0:
            return np.all(sum_col == half) and np.all(sum_lines == half)
        else:
            return np.all(np.isin(sum_col, (half, half+1))) and np.all(np.isin(sum_lines,(half, half+1)))


    def adjacent(self, state: TakuzuState):
        board = state.board.board
        v = np.lib.stride_tricks.sliding_window_view(board, 3, axis=1)
        v = v.reshape((v.shape[0]*v.shape[1],3)).sum(axis=1)
        rows = np.all(np.isin(v, (1, 2)))

        v = np.lib.stride_tricks.sliding_window_view(board, 3, axis=0)
        v = v.reshape((v.shape[0]*v.shape[1],3)).sum(axis=1)
        cols = np.all(np.isin(v, (1, 2)))

        return rows and cols

    def goal_test(self, state: TakuzuState):
        """Retorna True se e só se o estado passado como argumento é
        um estado objetivo. Deve verificar se todas as posições do tabuleiro
        estão preenchidas com uma sequência de números adjacentes."""

        if np.isin(state.board.board, 2):
            return False
        else:
            return self.half_half(state) and self.dif_rows_cols(state) and self.adjacent(state)



    def h(self, node: Node):
        """Função heuristica utilizada para a procura A*."""
        # TODO
        pass

    # TODO: outros metodos da classe


if __name__ == "__main__":
    board = Board.parse_instance_from_stdin()

    problem = Takuzu(board)

    goal_node = depth_first_tree_search(problem)

    print("Is goal?", problem.goal_test(goal_node.state))
    print("Solution:\n", goal_node.state.board)
