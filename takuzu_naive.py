# Grupo 30:
# 92759 Laura Quintas
# 92780 Raquel Romão

import time
from hashlib import new
from sys import stdin
from turtle import position
import numpy as np
from utils import (print_table, name)
from search import (
    Problem,
    Node,
    astar_search,
    breadth_first_tree_search,
    depth_first_tree_search,
    greedy_search,
    recursive_best_first_search,
    InstrumentedProblem, compare_searchers
)




class Board:
    """Representação interna de um tabuleiro de Takuzu.""" 

    def __init__(self, board, board_size, rows, cols): 
        self.board = board
        self.board_size = board_size
        #self.string = str(self.board)

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
        #state.last_action = (row,col,value)
        #state.changed_number = True
        self.rows[row, value] += 1
        self.cols[col,value] += 1
        
        #self.string = str(self.board.ravel()) # atualiza o hash value.
        

    def get_number(self, row: int, col: int):
        """Devolve o valor na respetiva posição do tabuleiro."""
        return self.board[row, col] 


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


    def adjacent_horizontal_numbers(self, row: int, col: int):
        """Devolve os valores imediatamente à esquerda e à direita,
        respectivamente."""
      
        if col == 0:
            return (self.get_number(row, col + 1),)
        
        elif col == self.board_size - 1:
            return (self.get_number(row, col - 1),)

        else:
            return (self.get_number(row, col - 1), self.get_number(row, col + 1))




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


        new_board = Board(board, board_size, line, col)
        return new_board


class TakuzuState:
    state_id = 0

    def __init__(self, board: Board, rows, cols):
        self.board = board
        self.board_size = board.board_size
        self.np_board = board.board
        self.id = TakuzuState.state_id
        TakuzuState.state_id += 1
        self.last_action = None
        self.changed_number=True
        self.rows = rows
        self.cols = cols


    def completed_rows(self):
        self.rows = set(str(arr) for arr in self.np_board if 2 not in arr)


    def completed_cols(self):
        self.cols = set(str(arr) for arr in self.board.board.transpose() if 2 not in arr)


    def __lt__(self, other):
        return self.id < other.id



    def actions(self):
        actions = []
        empty = self.empty_positions()
        
        #for i in empty:
        #para uma posição de cada vez:
        if len(empty) != 0:
            i = empty[0]
            actions.insert(0, (i[0],i[1],0))
            actions.insert(0, (i[0],i[1],1))
            
        return actions


    def linhas_p_isolada(self,i,qual,test_row,position_actions,deu_naslinhas):
        onde_dois = np.argwhere(test_row==2)
        dois = np.count_nonzero(test_row ==2)
        if qual==0:
            a=0
            b=1
        else:
            a=1
            b=0
        
        if self.board_size > 3 and dois==2 and (onde_dois[0,0]+1)==onde_dois[1,0] and (b in self.board.adjacent_horizontal_numbers(i[0],onde_dois[0,0]) or b in self.board.adjacent_horizontal_numbers(i[0],onde_dois[1,0])):
            position_actions.remove((i[0],i[1],a))
            deu_naslinhas =True
        elif self.board_size > 4 and dois==3 and (onde_dois[0,0]+1)==onde_dois[1,0] and (onde_dois[0,0]+2)==onde_dois[2,0]:
            position_actions.remove((i[0],i[1],a))
            deu_naslinhas =True
        elif self.board_size > 5 and dois==4 and (onde_dois[0,0]+1)==onde_dois[1,0] and (onde_dois[0,0]+2)==onde_dois[2,0] and (onde_dois[0,0]+3)==onde_dois[3,0]:
            position_actions.remove((i[0],i[1],a))
            deu_naslinhas =True
        elif self.board_size > 6 and dois==5 and (onde_dois[0,0]+1)==onde_dois[1,0] and (onde_dois[0,0]+2)==onde_dois[2,0] and (onde_dois[0,0]+3)==onde_dois[3,0] and (onde_dois[0,0]+4)==onde_dois[4,0]:
            position_actions.remove((i[0],i[1],a))
            deu_naslinhas =True
        
        return (position_actions, deu_naslinhas)


    def colunas_p_isolada(self, i, qual, test_col, position_actions, deu_nascolunas):
        onde_dois = np.argwhere(test_col==2)
        dois = np.count_nonzero(test_col ==2)
        if qual==0:
            a=0
            b=1
        else:
            a=1
            b=0

        if self.board_size > 3 and dois==2 and (onde_dois[0,0]+1)==onde_dois[1,0] and (b in self.board.adjacent_vertical_numbers(onde_dois[0,0],i[1]) or b in self.board.adjacent_vertical_numbers(onde_dois[1,0], i[1])):
            position_actions.remove((i[0],i[1],a))
            deu_nascolunas=True
        elif self.board_size > 4 and dois==3 and (onde_dois[0,0]+1)==onde_dois[1,0] and (onde_dois[0,0]+2)==onde_dois[2,0]:
            position_actions.remove((i[0],i[1],a))
            deu_nascolunas=True
        elif self.board_size > 5 and dois==4 and (onde_dois[0,0]+1)==onde_dois[1,0] and (onde_dois[0,0]+2)==onde_dois[2,0] and (onde_dois[0,0]+3)==onde_dois[3,0]:
            position_actions.remove((i[0],i[1],a))
            deu_nascolunas=True
        elif self.board_size > 6 and dois==5 and (onde_dois[0,0]+1)==onde_dois[1,0] and (onde_dois[0,0]+2)==onde_dois[2,0] and (onde_dois[0,0]+3)==onde_dois[3,0] and (onde_dois[0,0]+4)==onde_dois[4,0]:
            position_actions.remove((i[0],i[1],a))
            deu_nascolunas=True
        
        return (position_actions,deu_nascolunas)
        

    def para_linhas(self,i,qual,test_row):
        onde_dois = np.argwhere(test_row==2)
        dois = np.count_nonzero(test_row ==2)
        if qual==0:
            a=0
            b=1
        else:
            a=1
            b=0

        if self.board_size > 3 and dois==3 and (onde_dois[0,0]+1)==onde_dois[1,0] and (onde_dois[0,0]+2)==onde_dois[2,0]:
            if b in self.board.adjacent_horizontal_numbers(i[0],onde_dois[0,0]) and b in self.board.adjacent_horizontal_numbers(i[0],onde_dois[2,0]):
                self.board.set_number(i[0],onde_dois[1,0],a, self)
                self.board.set_number(i[0],onde_dois[0,0],b, self)
                self.board.set_number(i[0],onde_dois[2,0],b, self)


            elif b in self.board.adjacent_horizontal_numbers(i[0],onde_dois[0,0]):
                self.board.set_number(i[0],onde_dois[2,0],b, self)

            
            elif b in self.board.adjacent_horizontal_numbers(i[0],onde_dois[2,0]):
                self.board.set_number(i[0],onde_dois[0,0],b,self)

        
        elif self.board_size > 4 and dois==4 and (onde_dois[0,0]+1)==onde_dois[1,0] and (onde_dois[0,0]+2)==onde_dois[2,0] and (onde_dois[0,0]+3)==onde_dois[3,0]:
            if b in self.board.adjacent_horizontal_numbers(i[0],onde_dois[0,0]):
                self.board.set_number(i[0],onde_dois[0,0],b,self)
                self.board.set_number(i[0],onde_dois[1,0],a,self)
                self.board.set_number(i[0],onde_dois[2,0],b,self)
                self.board.set_number(i[0],onde_dois[3,0],b,self)


            elif b in self.board.adjacent_horizontal_numbers(i[0],onde_dois[3,0]):
                self.board.set_number(i[0],onde_dois[0,0],b,self)
                self.board.set_number(i[0],onde_dois[1,0],b,self)
                self.board.set_number(i[0],onde_dois[2,0],a,self)
                self.board.set_number(i[0],onde_dois[3,0],b,self)


            elif a in self.board.adjacent_horizontal_numbers(i[0],onde_dois[0,0]) or a in self.board.adjacent_horizontal_numbers(i[0],onde_dois[3,0]):
                self.board.set_number(i[0],onde_dois[0,0],b,self)
                self.board.set_number(i[0],onde_dois[3,0],b,self)


        elif self.board_size > 5 and dois==5 and (onde_dois[0,0]+1)==onde_dois[1,0] and (onde_dois[0,0]+2)==onde_dois[2,0] and (onde_dois[0,0]+3)==onde_dois[3,0] and (onde_dois[0,0]+4)==onde_dois[4,0]:
            if  a in self.board.adjacent_horizontal_numbers(i[0],onde_dois[0,0]) and a in self.board.adjacent_horizontal_numbers(i[0],onde_dois[4,0]):
                self.board.set_number(i[0],onde_dois[0,0],b,self)
                self.board.set_number(i[0],onde_dois[1,0],b,self)
                self.board.set_number(i[0],onde_dois[2,0],a,self)
                self.board.set_number(i[0],onde_dois[3,0],b,self)
                self.board.set_number(i[0],onde_dois[4,0],b,self)


    def para_colunas(self,i,qual, test_col):
        onde_dois = np.argwhere(test_col==2)
        dois = np.count_nonzero(test_col ==2)
        if qual==0:
            a=0
            b=1
        else:
            a=1
            b=0
        
        if self.board_size > 3 and dois==3 and (onde_dois[0,0]+1)==onde_dois[1,0] and (onde_dois[0,0]+2)==onde_dois[2,0]:
            if b in self.board.adjacent_vertical_numbers(onde_dois[0,0],i[1]) and b in self.board.adjacent_vertical_numbers(onde_dois[2,0],i[1]):
            
                self.board.set_number(onde_dois[1,0],i[1],a,self)
                self.board.set_number(onde_dois[0,0],i[1],b,self)
                self.board.set_number(onde_dois[2,0],i[1],b,self)


            elif b in self.board.adjacent_vertical_numbers(onde_dois[0,0],i[1]):
             
                self.board.set_number(onde_dois[2,0],i[1],b,self)

            
            elif b in self.board.adjacent_vertical_numbers(onde_dois[2,0], i[1]):
             
                self.board.set_number(onde_dois[0,0],i[1],b,self)

        
        elif self.board_size > 4 and dois==4 and (onde_dois[0,0]+1)==onde_dois[1,0] and (onde_dois[0,0]+2)==onde_dois[2,0] and (onde_dois[0,0]+3)==onde_dois[3,0]:
            if b in self.board.adjacent_vertical_numbers(onde_dois[0,0],i[1]):
       
                self.board.set_number(onde_dois[0,0],i[1],b,self)
                self.board.set_number(onde_dois[1,0],i[1],a,self)
                self.board.set_number(onde_dois[2,0],i[1],b,self)
                self.board.set_number(onde_dois[3,0],i[1],b,self)


            elif b in self.board.adjacent_vertical_numbers(onde_dois[3,0],i[1]):
        
                self.board.set_number(onde_dois[0,0],i[1],b,self)
                self.board.set_number(onde_dois[1,0],i[1],b,self)
                self.board.set_number(onde_dois[2,0],i[1],a,self)
                self.board.set_number(onde_dois[3,0],i[1],b,self)


            elif a in self.board.adjacent_vertical_numbers(onde_dois[0,0],i[1]) or a in self.board.adjacent_vertical_numbers(onde_dois[3,0],i[1]):
           
                self.board.set_number(onde_dois[0,0],i[1],b,self)
                self.board.set_number(onde_dois[3,0],i[1],b,self)


        elif self.board_size > 5 and dois==5 and (onde_dois[0,0]+1)==onde_dois[1,0] and (onde_dois[0,0]+2)==onde_dois[2,0] and (onde_dois[0,0]+3)==onde_dois[3,0] and (onde_dois[0,0]+4)==onde_dois[4,0]:
            if  a in self.board.adjacent_horizontal_numbers(onde_dois[0,0],i[1]) and a in self.board.adjacent_horizontal_numbers(onde_dois[4,0],i[1]):
         
                self.board.set_number(onde_dois[0,0],i[1],b,self)
                self.board.set_number(onde_dois[1,0],i[1],b,self)
                self.board.set_number(onde_dois[2,0],i[1],a,self)
                self.board.set_number(onde_dois[3,0],i[1],b,self)
                self.board.set_number(onde_dois[4,0],i[1],b,self)



    def empty_positions(self):
        result = np.where(self.board.board == 2)
        empty = np.column_stack(result)
        return empty


class Takuzu(Problem):
    def __init__(self, board: Board):
        """O construtor especifica o estado inicial."""
        self.initial = TakuzuState(board, None, None)
        #self.initial.completed_cols()
        #self.initial.completed_rows()
        self.visited_states = {}


    def actions(self, state: TakuzuState):
        """Retorna uma lista de ações que podem ser executadas a
        partir do estado passado como argumento."""
        actions = state.actions()
        return actions

    def hash(self, board):
        return str(board.flatten())

    def result(self, state: TakuzuState, action):
        """Retorna o estado resultante de executar a 'action' sobre
        'state' passado como argumento. A ação a executar deve ser uma
        das presentes na lista obtida pela execução de
        self.actions(state)."""
        
        new_board = state.board.copy()
        new_board.set_number(action[0], action[1], action[2])
        hash_state = self.hash(new_board.board)

        if hash_state in self.visited_states:
            return self.visited_states[hash_state]

        
        new_state = TakuzuState(new_board, None, None)

        
        
        self.visited_states[hash_state]= new_state
        
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

    def Window_linecol(self, arr):

        n = len(arr)
    
        window = arr[:3]
        a=True
        if all(window)==window[0]:
            a=False
        if a:
            for i in range(n - 3):
                window = window.pop(0) + arr[i + 3]
                if all(window==window[0]):
                    a=False
                    break
    
        return a



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
            
    
    def find_broken_rules(self, node: Node):
        board = node.state.board
        np_board = board.board
        np_board_t = np_board.transpose()
        board_size = board.board_size
        where = (node.action[0], node.action[1])
        row = np_board[where[0]]
        col = np_board[:,where[1]]

        if board_size%2==0:
            half = board_size//2
        else:
            half = board_size//2 + 1

        
        if np.any(board.rows[where[0]]>half) or np.any(board.cols[where[1]]>half):
            return board_size**3

        if not board.vertical(*node.action) or not board.horizontal(*node.action):
            return board_size**3
        
        if 2 not in row and np.any([np_board[i]==row for i in range(board_size) if i!=where[0]]): 
            return board_size**3

        if 2 not in col and np.any([np_board_t[i]==col for i in range(board_size) if i!=where[1]]):
            return board_size**3

        return 0


    def h1(self, node: Node): #tem que ter nome h!! senão n funciona
        """Função heuristica 1 utilizada para a procura A*. Além do numéro de casas vazias, é dada prioridade 
        a ações em linhas/colunas com poucos 2, para ser dado mais peso a serem completadas linhas/colunas."""
        f = 0
        twos = np.count_nonzero(node.state.board.board == 2)

        if node.parent:
            row_idx, col_idx,_ = node.action
            broken_rules = self.find_broken_rules(node)
            row = np.count_nonzero(node.state.board.board[row_idx] == 2)
            col = np.count_nonzero(node.state.board.board[:,col_idx] == 2)
            f = twos + broken_rules + row + col 
            #cheguei à conclusão que ter no row e col *2 ou outro era igual

        return f

    def h2(self, node: Node):
        """Função heuristica 2 utilizada para a procura A*. Além do numéro de casas vazias, tem-se em conta a
        diferença entre o número de 0's e 1's no tabuleiro."""
        
        f = 0
        twos = np.count_nonzero(node.state.board.board == 2)

        if node.parent:
            broken_rules = self.find_broken_rules(node)
            dif = abs((node.state.board.board==0).sum() - (node.state.board.board==1).sum())
            f = twos + dif + broken_rules
        
        return  f

    def h(self, node: Node): #melhor combinação até agora
        """Função heuristica 2 utilizada para a procura A*. Combinação das duas heurísticas acima."""
        
        f = 0
        twos = np.count_nonzero(node.state.board.board == 2)

        if node.parent:
            row_idx, col_idx,_ = node.action
            broken_rules = self.find_broken_rules(node)
            row = np.count_nonzero(node.state.board.board[row_idx] == 2)
            col = np.count_nonzero(node.state.board.board[:,col_idx] == 2)
            dif = abs((node.state.board.board==0).sum() - (node.state.board.board==1).sum())
            f = twos + dif + broken_rules + row + col

            if np.count_nonzero(node.state.board.board == 2) < node.state.board.board_size:
              f = f - node.state.board.board_size
        
        return  f


def compare_searchers(problem, header, searchers):
    
    def do(searcher, problem):
        p = InstrumentedProblem(problem)
        start_time = time.time()
        searcher(p)
        return p , "%s seconds" % (time.time() - start_time), "%s ms" % (time.time()*10**3 - start_time*10**3)


    table = [[name(s)] + [do(s, problem)] for s in searchers]
    print_table(table, header)


if __name__ == "__main__":
    
    board = Board.parse_instance_from_stdin()


    # Criar uma instância de Takuzu:
    problem = Takuzu(board)

    # Obter o nó solução usando a procura em profundidade:
    #goal_node = depth_first_tree_search(problem)
    
    #print(goal_node.state.board)

    compare_searchers(problem, header=['Searcher', 'selfsuccs/Goal tests/States/Time(s)/Time(ms)'], 
    searchers=[astar_search, breadth_first_tree_search, depth_first_tree_search, greedy_search])

#quanto se vai à função action self.succs +=1, succs=succesors?
#última coluna da tabela fica estranha  (<__m)-> é suposto ser str(self.found)[:4] , sendo self.found o state do resultado final
#<__main__.TakuzuState object at 0x7f3983f2e7d0> -> faz sentido
