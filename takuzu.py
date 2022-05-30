# takuzu.py: Template para implementação do projeto de Inteligência Artificial 2021/2022.
# Devem alterar as classes e funções neste ficheiro de acordo com as instruções do enunciado.
# Além das funções e classes já definidas, podem acrescentar outras que considerem pertinentes.

# Grupo 00:
# 92759 Laura Quintas
# 92780 Raquel Romão

#import sys (como estava antes)
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
)
import copy


class TakuzuState:
    state_id = 0

    def __init__(self, board):
        self.board = board
        self.id = TakuzuState.state_id
        TakuzuState.state_id += 1

    def __lt__(self, other):
        return self.id < other.id

    # TODO: outros metodos da classe


class Board:
    """Representação interna de um tabuleiro de Takuzu.""" 

    def __init__(self, board_size): 
        self.board = np.ones((board_size,board_size), dtype=int) 
        self.board_size = board_size
        self.info = np.zeros((board_size * 2,2), dtype=object) 
        
    
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
        self.board[row,col] = value
        if value == 1:
            self.info[row][1] +=1
            self.info[self.board_size + col][1] += 1
        else:
            self.info[row][0] +=1
            self.info[self.board_size + col][0] += 1
        

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

    @staticmethod
    def parse_instance_from_stdin():
        """Lê o test do standard input (stdin) que é passado como argumento
        e retorna uma instância da classe Board.

        Por exemplo:
            $ python3 takuzu.py < input_T01

            > from sys import stdin
            > stdin.readline()
        """
        #formato input:
        #4\n
        #2\t1\t2\t0\n
        #2\t2\t0\t2\n
        #2\t0\t2\t2\n
        #1\t1\t2\t0\n

        board_size = int(stdin.readline().rstrip('\n'))
        board = Board(board_size)

        for i in range(board_size):
            values = stdin.readline().strip('\n').split('\t') 
            for j in range(board_size):
                value= int(values[j])
                board.set_number(i,j,value)
                if value == 1:
                    board.info[i][1] +=1
                    board.info[board_size + j - 1][1] += 1                    
                elif value == 0:
                    board.info[i][0]+=1
                    board.info[board_size + j - 1][0] += 1 
               
        return board

    # TODO: outros metodos da classe


class Takuzu(Problem):
    def __init__(self, board: Board):
        """O construtor especifica o estado inicial."""
        #self.empty = np.array(list(zip(*np.where(board==2))))
        #self.states = np.array(TakuzuState(board))
        #self.initial_state = TakuzuState(board)

    def actions(self, state: TakuzuState):
        """Retorna uma lista de ações que podem ser executadas a
        partir do estado passado como argumento."""

        result = np.where(state.board.board == 2)
        empty = list(zip(result[0],result[1]))
        
        empty_arr = []
        for i in empty:
            empty_arr += [(i[0],i[1],0),(i[0],i[1],1)]
        return empty_arr


    def result(self, state: TakuzuState, action):
        """Retorna o estado resultante de executar a 'action' sobre
        'state' passado como argumento. A ação a executar deve ser uma
        das presentes na lista obtida pela execução de
        self.actions(state)."""
        new_board = copy.deepcopy(state.board)
        new_board.set_number(action[0],action[1],action[2])

        new_state = TakuzuState(new_board)  

        return new_state


    def dif_rows_cols(self, state: TakuzuState):
        _, row_counts = np.unique(state.board.board, axis=0, return_counts=True)
        unique_rows = len(row_counts) == state.board.board_size

        _, col_counts = np.unique(state.board.board, axis=1, return_counts=True)
        unique_cols = len(col_counts) == state.board.board_size

        return unique_rows and unique_cols

    
    '''
    def equal_number(self, state: TakuzuState, cord, ax): #ax=1 para linhas e ax=0 para colunas
        "Função auxiliar que verifica, retornando True ou False, se há um número igual de 0s e 1s, para uma determinada linha (ax=1) ou coluna (ax=0)."
        board_size = state.board.board_size
        equal = False
        if board_size % 2 == 0:
            if np.sum(state.board.board, axis = ax)[cord] == board_size//2:
                equal = True
        else:
            if np.sum(state.board.board, axis = ax)[cord] in [board_size//2 - 1, board_size//2 + 1] : #pode ser +1 ou -1 
                equal = True 
        return equal
    
    def equal_number_row(self, state: TakuzuState):
        "Função auxiliar que determina se há um número igual de 0s e 1s nas totalidade das linhas."
        board_size = state.board.board_size
        equal_test =[]
        for cord in (board_size - 1):
            equal_test += self.equal_number(state, cord, 1)

        return np.all(np.array(equal_test))

    def equal_number_col(self, state: TakuzuState):
        "Função auxiliar que determina se há um número igual de 0s e 1s nas totalidade das colunas."
        board_size = state.board.board_size
        equal_test =[]
        for cord in (board_size - 1):
            equal_test += self.equal_number(state, cord, 0)
        return np.all(np.array(equal_test))

    '''

    def half_half(self, state: TakuzuState):
        board_size= state.board.board_size
        half = board_size //2
        sum_col=np.sum(state.board.board, axis=0)
        sum_lines = np.sum(state.board.board, axis=1)
        if board_size % 2 == 0:
            return np.all(sum_col==half) and np.all(sum_lines==half) 
        else:
            return (np.all(sum_col==half) or np.all(sum_col==half-1)) and (np.all(sum_lines==half) or np.all(sum_lines==half-1))

    def adjacent(self, state: TakuzuState):
        board=state.board
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
            return self.half_half(state) and self.dif_rows_cols(state) and self.adjacent(state) #já ta
                



    def h(self, node: Node):
        """Função heuristica utilizada para a procura A*."""
        # TODO
        pass

    # TODO: outros metodos da classe


if __name__ == "__main__":
    # TODO:
    # Ler o ficheiro do standard input,
    # Usar uma técnica de procura para resolver a instância,
    # Retirar a solução a partir do nó resultante,
    # Imprimir para o standard output no formato indicado.
    pass

board = Board.parse_instance_from_stdin()
print(board)



problem= Takuzu(board)


goal_node = depth_first_tree_search(problem)

print("Is goal?", problem.goal_test(goal_node.state))
print("Solution:\n", goal_node.state.board)