# takuzu.py: Template para implementação do projeto de Inteligência Artificial 2021/2022.
# Devem alterar as classes e funções neste ficheiro de acordo com as instruções do enunciado.
# Além das funções e classes já definidas, podem acrescentar outras que considerem pertinentes.

# Grupo 00:
# 00000 Laura Quintas
# 92780 Raquel Romão

#import sys (como estava antes)
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

#olá

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
    """Representação interna de um tabuleiro de Takuzu.""" #estou a pensar numa lista de listas (array de arrays)

    def __init__(self, board_size):
        self.board = np.empty((board_size, board_size), dtype=object)  #None nas várias posições
        self.board_size = board_size

    def set_number(self, value, row: int, col: int): #adicionei para já esta função
        self.board[row,col] = value

    def get_number(self, row: int, col: int) -> int:
        """Devolve o valor na respetiva posição do tabuleiro."""
        return self.board[row, col] #como está agora pode ser devolvido None, o que pode dar problemas à frente 

    def adjacent_vertical_numbers(self, row: int, col: int) -> (int, int):
        """Devolve os valores imediatamente abaixo e acima,
        respectivamente."""
        #estar na coluna 0 ou na coluna n/[-1] é que vão ser aqui o issue
        if row == 0:
            return (None, self.board.get_number(row + 1, col))
        
        elif row == self.board_size - 1:
            return (self.board.get_number(row - 1, col), None)

        else:
            return (self.board.get_number(row - 1, col), self.board.get_number(row + 1, col))


    def adjacent_horizontal_numbers(self, row: int, col: int) -> (int, int):
        """Devolve os valores imediatamente à esquerda e à direita,
        respectivamente."""
        #estar na primeira ou última coluna é que vão ser o issue
        if col == 0:
            return (None, self.board.get_number(row, col + 1))
        
        elif col == self.board_size - 1:
            return (self.board.get_number(row, col - 1), None)

        else:
            return (self.board.get_number(row, col - 1), self.board.get_number(row, col + 1))

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

        #open and close file? ainda não testei se isto funciona, especialmente o line in stdin

        board_size = stdin.readline(0)
        board = Board(board_size)
       
        row=0
        for line in stdin:
            values = line.strip().split('\t') #retorna lista de strings com os numeros
            values = np.array(list(map(int, values)))
            values = np.where(values==2, None, values)

            for col in values:
                board.set_number(row, col) #precisa de otimização! pensei em adicionar a linha na totalidade também
                #adiciona os None por cima dos antigos, não adoro
            row+=1

        return board

    # TODO: outros metodos da classe


class Takuzu(Problem):
    def __init__(self, board: Board):
        """O construtor especifica o estado inicial."""
        # TODO
        pass

    def actions(self, state: TakuzuState):
        """Retorna uma lista de ações que podem ser executadas a
        partir do estado passado como argumento."""
        # TODO
        pass

    def result(self, state: TakuzuState, action):
        """Retorna o estado resultante de executar a 'action' sobre
        'state' passado como argumento. A ação a executar deve ser uma
        das presentes na lista obtida pela execução de
        self.actions(state)."""
        # TODO
        pass

    def goal_test(self, state: TakuzuState):
        """Retorna True se e só se o estado passado como argumento é
        um estado objetivo. Deve verificar se todas as posições do tabuleiro
        estão preenchidas com uma sequência de números adjacentes."""
        # TODO
        pass

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
