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
        self.board = np.ones((board_size,board_size), dtype=object) 
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
        elif value == 0:
            self.info[row][0] +=1
            self.info[self.board_size + col][0] += 1
        #else? tipo vamos usar o set_number para voltar a pôr como vazio = 2 ou aquilo simplesmente tem guardado uma board e volta atrás assim?

    def get_number(self, row: int, col: int) -> int:
        """Devolve o valor na respetiva posição do tabuleiro."""
        return self.board[row, col] #como está agora pode ser devolvido None, o que pode dar problemas à frente 

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
        self.initial_state = TakuzuState(board)

    def actions(self, state: TakuzuState):
        """Retorna uma lista de ações que podem ser executadas a
        partir do estado passado como argumento."""

        result = np.where(state.board.board == 2)
        print(result)
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
        new_board = state.board
        new_board.set_number(action[0],action[1],action[2])

        new_state = TakuzuState(new_board)  #nao sei se nao vamos ter de pôr (action[0], action[1], action[2]) probably

        return new_state

    def goal_test(self, state: TakuzuState):
        """Retorna True se e só se o estado passado como argumento é
        um estado objetivo. Deve verificar se todas as posições do tabuleiro
        estão preenchidas com uma sequência de números adjacentes."""
        # TODO
        pass

    #se where == 2 não for lista vazia (? ver), então não temos solução
    #fazer função auxiliar para ver se as linhas estão ok e usar transposta dessa para as colunas

    #sum = n/2 (se não for impar) para respeitar numero igual de 1s e 0s; para impar fazer outro if

    #

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
print(type(board.board[0]))

print(board.adjacent_vertical_numbers(3, 3))
print(board.adjacent_horizontal_numbers(3, 3))

problem= Takuzu(board)

initial_state = TakuzuState(board)

print(initial_state.board.get_number(0, 0))

result_state = problem.result(initial_state, (0, 0, 0))

print(result_state.board.get_number(0, 0))
print(initial_state.board)
print(type(initial_state.board))
print(type(initial_state.board[1]))


print(problem.actions(initial_state))

print(problem.actions(result_state))