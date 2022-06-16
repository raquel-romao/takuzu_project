# Grupo 30:
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
    InstrumentedProblem,
)


class TakuzuState:
    state_id = 0

    def __init__(self, board):
        self.board = board
        self.id = TakuzuState.state_id
        TakuzuState.state_id += 1
        self.open = False
        self.possible_actions = None


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
                    position_actions =[]

                    if line[i[0]][0] < half and col[i[1]][0] < half and self.board.adjacent_vertical_numbers(i[0],i[1]).count(0)!=2 and self.board.adjacent_horizontal_numbers(i[0],i[1]).count(0)!=2:
                        position_actions.append((i[0],i[1],0))

                    if line[i[0]][1] < half and col[i[1]][1] < half and self.board.adjacent_vertical_numbers(i[0],i[1]).count(1)!=2 and self.board.adjacent_horizontal_numbers(i[0],i[1]).count(1)!=2:
                        position_actions.append((i[0],i[1],1))
                    
                    if len(position_actions)==2:
                        actions.append(position_actions[0])
                        actions.append(position_actions[1])
                    
                    elif len(position_actions)==1:
                        actions.insert(0,position_actions[0])

                    else:
                        self.possible_actions = []
                        return self.possible_actions

                self.possible_actions = actions
    
            return self.possible_actions
        else:
            return []


    
    def expand(self):
        self.open = True
    


class Board:
    """Representação interna de um tabuleiro de Takuzu.""" 

    def __init__(self, board, board_size, empty): 
        self.board = board
        self.board_size = board_size
        self.empty = empty
        self.string = str(self.board.ravel())
        
    
    def __str__(self):
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
        self.empty.delete((row,col))
        self.string = str(self.board.ravel) # atualiza o hash value.
        
        
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
        new_board = self.board.copy()
        return Board(new_board, self.board_size, self.empty)



    @staticmethod
    def parse_instance_from_stdin():
        """Lê o test do standard input (stdin) que é passado como argumento
        e retorna uma instância da classe Board.
        """

        board_size = int(stdin.readline().rstrip('\n'))
        board = np.ones((board_size, board_size), dtype=int)
        empty = np.array([])
        for i in range(board_size):
            values = stdin.readline().strip('\n').split('\t') 
            for j in range(board_size):
                value = int(values[j])
                board[i, j] = value
                if value ==2:
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

        #avoid creating same state, helps with space
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
        half = board_size //2
        sum_col = np.sum(state.board.board, axis=0)
        sum_lines = np.sum(state.board.board, axis=1)
        if board_size % 2 == 0:
            return np.all(sum_col==half) and np.all(sum_lines==half) 
        else:
            col = np.where(sum_col == half+1, half, sum_col)
            lin = np.where(sum_lines == half+1, half, sum_lines)
            return np.all(col==half) and np.all(lin==half)

    def adjacent(self, state: TakuzuState): #podemos otimizar visto que nao precisamos de ver adjacentes verticais para a primeira e ultima linha e nao precisamos de ver adjacentes horizontais para a primeira e ultima coluna
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

    
    def find_broken_rules(self, node: Node, board_np, i):
        board = node.state.board
        board_size = board.board_size

        indices = np.arange(board_size)
        
        if 2 not in board_np[i, :]: 
                
            if np.any(board_np[indices != i, :] == board_np[i, :]):
                return board_size**3

        return 0

    def count_filled(self, board_np):
        count_filled = 0
        
        for line in board_np:
            if 2 not in line:
                count_filled += 1

        return count_filled


    def h(self, node: Node):
        """Função heuristica utilizada para a procura A*."""

        current_state = node.state
        parent_node = node.parent
        last_action = node.action
        board = node.state.board
        board_np = node.state.board.board
        board_size = board.board_size

        f = 0

        if self.goal_test(current_state):
            return 0

        number_actions = len(current_state.actions())
        if number_actions == 0:
            return board_size**3
        
        broken_rule = 0
        if parent_node != None:
            parent_state = parent_node.state
            lin_changed = last_action[0]
            col_changed = last_action[1]
            #val_inserted = last_action[2]

            broken_rule = self.find_broken_rules(node, board_np, lin_changed)

            if broken_rule!=0:
                return broken_rule
            
            broken_rule = self.find_broken_rules(node, np.transpose(board_np), col_changed)

            if broken_rule!=0:
                return broken_rule

            f += parent_state.possible_actions.index(last_action)

            
        f += board_size - self.count_filled(board_np)
        f += board_size - self.count_filled(np.transpose(board_np))

        return f 


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




