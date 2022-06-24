
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




class Board:
    """Representação interna de um tabuleiro de Takuzu.""" 

    def __init__(self, board, board_size, rows, cols): 
        self.board = board
        self.board_size = board_size
        self.string = str(self.board)

        self.rows = rows
        self.cols = cols
        
    
    def __str__(self):
        prettyprint = ''
        for i in self.board:
            for j in range(self.board_size):
                if j == self.board_size-1:
                    prettyprint += f'{i[j]}\n'
                else:
                    prettyprint += f'{i[j]}\t'
        return prettyprint.rstrip('\n')

    def get_board(self):
        return self.board

    def set_number(self, row: int, col: int, value):
        self.board[row, col] = value
        self.rows[row, value] += 1
        self.cols[col,value] += 1
        self.string = str(self.board.ravel()) # atualiza o hash value.
        

    def get_number(self, row: int, col: int):
        """Devolve o valor na respetiva posição do tabuleiro."""
        return self.board[row, col] 


    """def count(self, t: tuple, i: int):
        return sum(x == i for x in t)"""

    '''def adjacent_vertical_numbers(self, row: int, col: int):
        """Devolve os valores imediatamente abaixo e acima,
        respectivamente."""

        if row == 0:
            return (self.get_number(row + 1, col),)
        
        elif row == self.board_size - 1:
            return (self.get_number(row - 1, col),)

        else:
            return (self.get_number(row - 1, col), self.get_number(row + 1, col))'''



    def horizontal(self, row: int, col: int, move: int):
        
        n = self.board_size
        check = []

        if (col not in (n-1, n-2)):
            check.append((self.get_number(row, col+1), self.get_number(row, col+2))) #guardar array de posições contíguas
        if (col not in (0, 1)):
            check.append((self.get_number(row, col-2), self.get_number(row, col-1)))
        if (col not in (0, n-1)):
            check.append((self.get_number(row, col-1), self.get_number(row, col+1)))

  

        return all(t.count(move) != 2 for t in check)


    def vertical(self, row: int, col:int, move:int):
        n = self.board_size
        check = []

        if (row not in (n-1, n-2)):
            check.append((self.get_number(row+1, col), self.get_number(row+2, col)))
        if (row not in (0, 1)):
            check.append((self.get_number(row-1, col), self.get_number(row-2, col)))
        if (row not in (0, n-1)):
            check.append((self.get_number(row-1, col), self.get_number(row+1, col)))


        return all(t.count(move) != 2 for t in check)

    '''def adjacent_horizontal_numbers(self, row: int, col: int):
        """Devolve os valores imediatamente à esquerda e à direita,
        respectivamente."""
      
        if col == 0:
            return (self.get_number(row, col + 1),)
        
        elif col == self.board_size - 1:
            return (self.get_number(row, col - 1),)

        else:
            return (self.get_number(row, col - 1), self.get_number(row, col + 1))'''


    def __hash__(self):
        return hash(self.string)


    def copy(self):
        new_board = self.board.copy()
        new_line = self.rows.copy()
        new_col = self.cols.copy()
        return Board(new_board, self.board_size,  new_line, new_col)



    @staticmethod
    def parse_instance_from_stdin():
        """Lê o test do standard input (stdin) que é passado como argumento
        e retorna uma instância da classe Board.
        """

        board_size = int(stdin.readline().rstrip('\n'))
        board = np.ones((board_size, board_size), dtype=int)


        for i in range(board_size):

            values = stdin.readline().strip('\n').split('\t') 
            for j in range(board_size):
                value = int(values[j])
                board[i, j] = value


        line = np.column_stack(((board==0).sum(axis=1), (board==1).sum(axis=1)))
        col = np.column_stack(((board==0).sum(axis=0), (board==1).sum(axis=0)))


        new_board = Board(board, board_size, line,col)
        return new_board


class TakuzuState:
    state_id = 0

    def __init__(self, board: Board, action):
        self.board = board
        self.board_size = board.board_size
        self.np_board = board.board
        self.id = TakuzuState.state_id
        TakuzuState.state_id += 1
        self.last_action = action
        #self.rows = set(str(arr) for arr in board.board)
        #self.cols = set(str(arr) for arr in board.board.transpose())


    def __lt__(self, other):
        return self.id < other.id


    def __hash__(self): 
        return hash(self.board)


    def actions(self):

        actions =[]

        if self.board_size % 2 == 0:
            half = self.board_size //2
        else:
            half = self.board_size //2 + 1
        
        if self.last_action!=None:
            if np.any(self.board.rows[self.last_action[0]] > half) or np.any(self.board.cols[self.last_action[1]] > half) or not self.board.horizontal(self.last_action[0],self.last_action[1],self.last_action[2]) or not self.board.vertical(self.last_action[0],self.last_action[1],self.last_action[2]):

                return actions


        empty = self.empty_positions()
        for i in empty:

            row_idx, col_idx = i
            position_actions = []


            if self.board.rows[row_idx, 0] < half and self.board.cols[col_idx, 0] < half and self.board.horizontal(row_idx, col_idx, 0) and self.board.vertical(row_idx, col_idx, 0):
                position_actions.append((row_idx, col_idx, 0))


            if self.board.rows[row_idx, 1] < half and self.board.cols[col_idx, 1] < half and self.board.horizontal(row_idx, col_idx, 1) and self.board.vertical(row_idx, col_idx, 1):
                position_actions.append((row_idx, col_idx, 1))

            '''
            for a in position_actions:
                test_row = self.board.board[a[0]].copy()
                test_row[a[1]] = a[2] 
                test_col = self.board.board.transpose()[:,a[1]].copy()
                test_col[a[0]] = a[2]

                if str(test_row) in self.rows or str(test_col) in self.cols:
                    position_actions.remove(a)

                if 2 not in test_row and str(test_row) not in self.rows:
                    self.rows.add(str(test_row))
                if 2 not in test_col and str(test_col) not in self.cols:
                    self.cols.add(str(test_col))
            '''

            if len(position_actions)==2:
                actions.insert(0, position_actions[0])
                actions.insert(0, position_actions[1])

            elif len(position_actions)==1:
                a=position_actions[0]
                self.board.set_number(a[0],a[1],a[2])


                if len(actions)==0 and 2 not in self.board.board:
                    actions.append(a)
                    self.board.rows[a[0],a[2]] -=1
                    self.board.cols[a[1],a[2]] -=1

            else:
                actions = []
                return actions

        

        return actions


    '''def check_line(self, half):
        line = self.board.board[self.last_action[0]]
        v = np.lib.stride_tricks.sliding_window_view(line, 3)

        return np.any(self.board.rows[self.last_action[0]] > half) or any(np.all(a==a[0]) for a in v)

    def check_col(self, half):
        col = self.board.board[:,self.last_action[1]]
        v = np.lib.stride_tricks.sliding_window_view(col, 3)

        return np.any(self.board.cols[self.last_action[1]] > half) or any(np.all(a==a[0]) for a in v)'''

    def empty_positions(self):
        result = np.where(self.board.board == 2)
        empty = np.column_stack(result)
        return empty
    


class Takuzu(Problem):
    def __init__(self, board: Board):
        """O construtor especifica o estado inicial."""
        self.initial = TakuzuState(board, None)
        self.visited_states = {}


    def actions(self, state: TakuzuState):
        """Retorna uma lista de ações que podem ser executadas a
        partir do estado passado como argumento."""
        actions = state.actions()
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

        new_state = TakuzuState(new_board, action)
        self.visited_states.update({hash_state: new_state})
        
        return new_state


    def dif_rows_cols(self, state: TakuzuState):
        _, row_counts = np.unique(state.board.board, axis=0, return_counts=True)
        unique_rows = len(row_counts) == state.board.board_size

        _, col_counts = np.unique(state.board.board, axis=1, return_counts=True)
        unique_cols = len(col_counts) == state.board.board_size

        return unique_rows and unique_cols


    def half_half(self, state: TakuzuState):
        half = state.board_size //2
    
        if state.board_size % 2 == 0:
            return np.all(state.board.rows == half) and np.all(state.board.cols == half)
        else:
            return np.all(np.isin(state.board.rows, [half, half +1])) and np.all(np.isin(state.board.cols, [half, half+1]))


    def Window_Sum(self, arr):

        n = len(arr)
    
        window_sum = sum(arr[:3])
        a=True
        if window_sum not in [1,2]:
            a=False
        if a:
            for i in range(n - 3):
                window_sum = window_sum - arr[i] + arr[i + 3]
                if window_sum not in [1,2]:
                    a=False
                    break
    
        return a

    '''def adjacent(self, state: TakuzuState):
        board = state.board.board
        v = np.lib.stride_tricks.sliding_window_view(board, 3, axis=1)
        v = v.reshape((v.shape[0]*v.shape[1],3)).sum(axis=1)
        rows = np.all(np.isin(v, (1, 2)))

        v = np.lib.stride_tricks.sliding_window_view(board, 3, axis=0)
        v = v.reshape((v.shape[0]*v.shape[1],3)).sum(axis=1)
        cols = np.all(np.isin(v, (1, 2)))

        return rows and cols'''

    def adjacent(self, state:TakuzuState):
        board = state.board.board 

        rows = all([self.Window_Sum(arr) for arr in board])
        cols = all(self.Window_Sum(arr) for arr in board.transpose())

        return rows and cols

    def goal_test(self, state: TakuzuState):
        """Retorna True se e só se o estado passado como argumento é
        um estado objetivo. Deve verificar se todas as posições do tabuleiro
        estão preenchidas com uma sequência de números adjacentes."""

        return 2 not in state.board.board and self.dif_rows_cols(state) and self.half_half(state) and self.adjacent(state)
            
    
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
    goal_node = astar_search(problem)
    # Verificar se foi atingida a solução
    #print("Is goal?", problem.goal_test(goal_node.state))
    print(goal_node.state.board)