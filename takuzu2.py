
# Grupo 30:
# 92759 Laura Quintas
# 92780 Raquel Romão

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
        self.rows = set(str(arr) for arr in board.board)
        self.cols = set(str(arr) for arr in board.board.transpose())


    def __lt__(self, other):
        return self.id < other.id


    def __hash__(self): 
        return hash(self.board)


    def actions(self):
        #if not self.open: #então tiramos? -> e expand + à frente?
        line = np.column_stack(((self.board.board==0).sum(axis=1), (self.board.board==1).sum(axis=1)))
        col = np.column_stack(((self.board.board==0).sum(axis=0), (self.board.board==1).sum(axis=0)))
        actions = []
        empty = self.empty_positions()

        if self.board.board_size % 2 == 0:
            half = self.board.board_size //2
        else:
            half = self.board.board_size //2 + 1

        for i in empty:
            row_idx, col_idx = i

            position_actions = []

            if line[row_idx][0] < half and col[col_idx][0] < half and self.board.horizontal(row_idx, col_idx, 0) and self.board.vertical(row_idx, col_idx, 0):
                position_actions.append((row_idx, col_idx, 0))


            if line[row_idx][1] < half and col[col_idx][1] < half and self.board.horizontal(row_idx, col_idx, 1) and self.board.vertical(row_idx, col_idx, 1):
                position_actions.append((row_idx, col_idx, 1))


            """

            if line[i[0]][0] < half and col[i[1]][0] < half and \
                self.board.adjacent_vertical_numbers(i[0],i[1]).count(0)!=2 and \
                    self.board.adjacent_horizontal_numbers(i[0],i[1]).count(0)!=2:
                position_actions.append((i[0],i[1],0))

            if line[i[0]][1] < half and col[i[1]][1] < half and \
                self.board.adjacent_vertical_numbers(i[0],i[1]).count(1)!=2 and \
                    self.board.adjacent_horizontal_numbers(i[0],i[1]).count(1)!=2:
                position_actions.append((i[0],i[1],1))
            """
            for a in position_actions:
                test_row = self.board.board[a[0]].copy()
                test_row[a[1]] = a[2] 
                test_col = self.board.board[:,a[1]].copy()
                test_col[a[0]] = a[2]

                if str(test_row) in self.rows or str(test_col) in self.cols:
                    position_actions.remove(a)

            if len(position_actions)==2:
                actions.append(position_actions[0])
                actions.append(position_actions[1])

            elif len(position_actions)==1:
                a=position_actions[0]
                self.board.set_number(*a)
                line[row_idx][a[2]] += 1
                col[col_idx][a[2]] += 1

            else:
                actions = []
                return actions

        if 2 not in self.board.board and len(actions)==0 and len(position_actions)!=0:
            actions.append(a)
        
        return actions

    def empty_positions(self):
        result = np.where(self.board.board == 2)
        empty = np.column_stack(result)
        return empty
    
    def expand(self):
        self.open = True
    

class Board:
    """Representação interna de um tabuleiro de Takuzu.""" 

    def __init__(self, board, board_size, empty): 
        self.board = board
        self.board_size = board_size
        self.empty = empty
        self.string = str(self.board)
        
    
    def __str__(self):
        prettyprint = ''
        for i in self.board:
            for j in range(len(i)):
                if j == len(i)-1:
                    prettyprint += f'{i[j]}\n'
                else:
                    prettyprint += f'{i[j]}\t'
        return prettyprint.rstrip('\n')

    def set_number(self, row: int, col: int, value): 
        self.board[row, col] = value
        self.string = str(self.board.ravel()) # atualiza o hash value.
        
    def get_number(self, row: int, col: int) -> int:
        """Devolve o valor na respetiva posição do tabuleiro."""
        return self.board[row, col] 


    def count(self, t: tuple, i: int):
        return sum(x == i for x in t)

    def adjacent_vertical_numbers(self, row: int, col: int):
        """Devolve os valores imediatamente abaixo e acima,
        respectivamente."""

        if row == 0:
            return (self.get_number(row + 1, col),)
        
        elif row == self.board_size - 1:
            return (self.get_number(row - 1, col),)

        else:
            return (self.get_number(row - 1, col), self.get_number(row + 1, col))



    def horizontal(self, row: int, col: int, move: int):
        
        n = self.board_size
        check = []

        if (col not in (n-1, n-2)):
            check.append((self.get_number(row, col+1), self.get_number(row, col+2))) #guardar array de posições contíguas
        if (col not in (0, 1)):
            check.append((self.get_number(row, col-1), self.get_number(row, col-2)))
        if (col not in (0, n-1)):
            check.append((self.get_number(row, col-1), self.get_number(row, col+1)))


        return all(self.count(t, move) != 2 for t in check)


    def vertical(self, row: int, col:int, move:int):
        n = self.board_size
        check = []

        if (row not in (n-1, n-2)):
            check.append((self.get_number(row+1, col), self.get_number(row+2, col)))
        if (row not in (0, 1)):
            check.append((self.get_number(row-1, col), self.get_number(row-2, col)))
        if (row not in (0, n-1)):
            check.append((self.get_number(row-1, col), self.get_number(row+1, col)))

        return all(self.count(t, move) != 2 for t in check)

    

    def adjacent_horizontal_numbers(self, row: int, col: int):
        """Devolve os valores imediatamente à esquerda e à direita,
        respectivamente."""
      
        if col == 0:
            return (self.get_number(row, col + 1),)
        
        elif col == self.board_size - 1:
            return (self.get_number(row, col - 1),)

        else:
            return (self.get_number(row, col - 1), self.get_number(row, col + 1))


    def __hash__(self):
        return hash(self.string)

    def copy(self):
        new_board = self.board.copy()
        return Board(new_board, self.board_size, self.empty)



    @staticmethod
    def parse_instance_from_stdin():
        """Lê o test do standard input (stdin) que é passado como argumento
        e retorna uma instância da classe Board.
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
        #state.expand()
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

        return 2 not in state.board.board
            
    
    def find_broken_rules(self, node: Node, board_np, i):
        board = node.state.board
        board_size = board.board_size

        indices = np.arange(board_size)
        
        if 2 not in board_np[i, :]: 
                
            if np.any(board_np[indices != i, :] == board_np[i, :]):
                return board_size**3

        return 0


    def h(self, node: Node):
        """Função heuristica utilizada para a procura A*."""
        
        twos = np.count_nonzero(node.state.board.board == 2)

        if node.parent:
            row_idx, col_idx,_ = node.action
            row = np.count_nonzero(node.state.board.board[row_idx] == 2)
            col = np.count_nonzero(node.state.board.board[:,col_idx] == 2)
            return twos + 2*row + 2*col #para prioritizar ações em linhas com poucos 2, para dar mais peso a completar a linha
        return twos

        #return np.count_nonzero(node.state.board.board == 2) #f #como estava antes mas decidi meter o número de casas vazias como h p experimentar, quero fechar a árvore o mais rápido possível e tentar primeiro os estados com menos casas

if __name__ == "__main__":
    
    board = Board.parse_instance_from_stdin()

    # Criar uma instância de Takuzu:
    problem = Takuzu(board)
    # Obter o nó solução usando a procura em profundidade:
    goal_node = depth_first_tree_search(problem)
    # Verificar se foi atingida a solução
    #print("Is goal?", problem.goal_test(goal_node.state))
    print(goal_node.state.board)


