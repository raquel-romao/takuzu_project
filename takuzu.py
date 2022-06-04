# Devem alterar as classes e funções neste ficheiro de acordo com as instruções do enunciado.
# Além das funções e classes já definidas, podem acrescentar outras que considerem pertinentes.

# Grupo 00:
# 92759 Laura Quintas
# 92780 Raquel Romão

#import sys (como estava antes)

import copy
from hashlib import new
from sys import stdin
import numpy as np
from search import (
    Problem,
    Node,
    astar_search,
    breadth_first_tree_search,
    depth_first_tree_search,
    greedy_search,
    recursive_best_first_search,
    InstrumentedProblem,
)


class TakuzuState:
    state_id = 0

    def __init__(self, board):
        self.board = board
        self.id = TakuzuState.state_id
        TakuzuState.state_id += 1
        self.possible_actions = None

    def __lt__(self, other):
        return self.id < other.id


    def __hash__(self): #também é para pôr aqui? -> acho que no need
        return hash(self.board)

    def actions(self):
        if self.possible_actions == None:
            line = list(zip((self.board.board==0).sum(axis=1), (self.board.board==1).sum(axis=1)))
            col = list(zip((self.board.board==0).sum(axis=0), (self.board.board==1).sum(axis=0)))
            actions = []
            empty = self.empty_positions()

            if self.board.board_size % 2 == 0:
                half = self.board.board_size //2
            else:
                half = self.board.board_size //2 +1

            for i in empty:
                if line[i[0]][0] < half and col[i[1]][0]< half:
                    actions.append((i[0],i[1],0))
                if line[i[0]][1] < half and col[i[1]][1] < half:
                    actions.append((i[0],i[1],1))

            self.possible_actions = actions
        
        return self.possible_actions

    def empty_positions(self):
        result = np.where(self.board.board == 2)
        empty = list(zip(result[0],result[1]))
        return empty
    
    def set_actions(self):
        self.possible_actions = []

    #quando gero um estado posso meter aqui qual a jogada que me fez chegar ao estado -> posso depois ver se na heurística foi quebrada alguma regra com esta jogada ou não

class Board:
    """Representação interna de um tabuleiro de Takuzu.""" 

    def __init__(self,board, board_size): 
        self.board = board
        self.board_size = board_size
        self.string = str(self.board.ravel())
        
    
    def __str__(self):
        prettyprint=''
        for i in self.board:
            for j in range(len(i)):
                if j == len(i)-1:
                    prettyprint += f'{i[j]}\n'
                else:
                    prettyprint += f'{i[j]}    '
        return prettyprint

    def set_number(self, row: int, col: int, value): 
        self.board[row, col] = value
        self.string = str(self.board.ravel)
        
        
    def get_number(self, row: int, col: int) -> int:
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
        new_board= self.board.copy()
        return Board(new_board, self.board_size)


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
        board = np.ones((board_size,board_size), dtype=int)

        for i in range(board_size):
            values = stdin.readline().strip('\n').split('\t') 
            for j in range(board_size):
                value = int(values[j])
                board[i, j]= value

        new_board=Board(board,board_size)

        return new_board


class Takuzu(Problem):
    def __init__(self, board: Board):
        """O construtor especifica o estado inicial."""
        self.initial = TakuzuState(board)
        self.visited_states = {}


    def actions(self, state: TakuzuState):
        """Retorna uma lista de ações que podem ser executadas a
        partir do estado passado como argumento."""

        return state.actions()


    def result(self, state: TakuzuState, action):
        """Retorna o estado resultante de executar a 'action' sobre
        'state' passado como argumento. A ação a executar deve ser uma
        das presentes na lista obtida pela execução de
        self.actions(state)."""
        
        new_board = state.board.copy()

        new_board.set_number(action[0], action[1], action[2])

        hash_state = hash(new_board)

        #avoid creating same state, helps with space
        if hash_state in self.visited_states:
            #avoids going through a path that was already visited
            self.visited_states[hash_state].set_actions()
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
        half = board_size //2
        sum_col = np.sum(state.board.board, axis=0)
        sum_lines = np.sum(state.board.board, axis=1)
        if board_size % 2 == 0:
            return np.all(sum_col==half) and np.all(sum_lines==half) 
        else:
            col = np.where(sum_col == half+1, half, sum_col)
            lin = np.where(sum_lines == half+1, half, sum_lines)
            return np.all(col==half) and np.all(lin==half)

    def adjacent(self, state: TakuzuState):
        board = state.board
        for i in range(board.board_size):
            for j in range(board.board_size):
                if board.adjacent_vertical_numbers(i,j).count(board.get_number(i,j))==2 or board.adjacent_horizontal_numbers(i,j).count(board.get_number(i,j))==2:
                    return False
        return True


    def goal_test(self, state: TakuzuState):
        """Retorna True se e só se o estado passado como argumento é
        um estado objetivo. Deve verificar se todas as posições do tabuleiro
        estão preenchidas com uma sequência de números adjacentes."""

        if 2 in state.board.board:
            return False
        else:
            return self.half_half(state) and self.dif_rows_cols(state) and self.adjacent(state)




    def h(self, node: Node): #for each state we have to define an heuristic
        """Função heuristica utilizada para a procura A*."""

        current_state = node.state
        #com node.parent consigo aceder ao nó pai e comparar o que mudou -> ou então adicionar função à classe State que me dá a última jogada

        #g = node.state.id #pensei usar o id do node mas não é apenas isto -> not sure até que ponto é necessário somar -> ya acho que é só mesmo para devolver f


        if self.goal_test(current_state):
            f = 0

        elif not self.half_half(current_state) or not self.dif_rows_cols(current_state) or not self.adjacent(current_state):
            f = 10
        
        #h = f + g (g = shortest path from the start node to node n, acho que isto é path cost, path vai ter sempre o mesmo tamanho (preencher as caixas vazias)) -> acho que no need, apenas f suficiente
        


        #ideias para heurísticas:
        #devolver 0 se não violar nenhuma regra
        #aumentat 10 ou assim por cada regra que se viola
        #devolver um numero bue alto para jogadas inválidas 
        #MAS por ex se faltarem muitas peças para adicionar numa linha por exemplo e tivermos bue longe do n//2, jogar um 1 seria mais relavante, devolver 0 no caso de jogar 1 (o ideal) ou devolver 1 no caso de jogar 0 (pode ajudar mas não muito)
    
    # TODO: outros metodos da classe

if __name__ == "__main__":
    # $ python3 takuzu < i1.txt
    board = Board.parse_instance_from_stdin()
    print(board)
    # Criar uma instância de Takuzu:
    problem = Takuzu(board)
    # Obter o nó solução usando a procura em profundidade:
    goal_node = depth_first_tree_search(problem)
    # Verificar se foi atingida a solução
    print("Is goal?", problem.goal_test(goal_node.state))
    print("Solution:\n", goal_node.state.board)




