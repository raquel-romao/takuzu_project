**TAKUZU SOLVER - Artificial Intelligence Project 2021/22 (P4)**

**Introduction**

This project for the Artificial Intelligence (AI) course aims to develop a Python program that solves the Takuzu problem using AI search techniques.

**Problem Description**

The Takuzu problem, also known as Binairo, Binary Puzzle, Binary Sudoku, or Tic-Tac-Logic, is a logic game with two possible origins, both in 2009: Tohu wa Vohu was invented by the Italian Adolfo Zanellati, and Binairo was created by the Belgians Peter De Schepper and Frank Coussement.

The Takuzu game takes place on a square grid board. Each cell in the grid can contain the numbers 0 or 1.

Given a board with an N×N grid, partially filled with 0s and 1s, the objective of Takuzu is to fill the entire grid with 0s and 1s such that:

- There is an equal number of 1s and 0s in each row and column (or one more for grids with an odd dimension)
- No more than two identical numbers are adjacent to each other (horizontally or vertically)
- All rows are different
- All columns are different
- The contents of the initially filled grid positions cannot be changed.

**Objective**

The objective of this project is to develop a Python 3.8 program that, given an instance of Takuzu, returns a solution, i.e., a fully filled grid.

**Input Format**

The input files (.txt) represent instances of the Takuzu problem and follow this format:

- The first line contains a single integer N, indicating the size of the N × N grid.
- The following N lines indicate the contents of each of the N rows of the grid. An empty position is represented by the number 2.
- Each line ends with \n, and the columns are separated by \t.

Example of input:

4\n

2\t1\t2\t0\n

2\t2\t0\t2\n

2\t0\t2\t2\n

1\t1\t2\t0\n

**Output Format**

The program's output should describe a solution to the Takuzu problem described in the input file, i.e., a fully filled grid with 0s and 1s that adheres to the previously stated rules. The output should follow this format:

- Each of the N lines indicates the content of each of the N rows of the grid.
- Both the rows and columns appear in increasing order.
- All lines, including the last one, are terminated by the newline character, i.e., \n.

Example of output:

0\t1\t1\t0\n

1\t0\t0\t1\n

0\t0\t1\t1\n

1\t1\t0\t0\n

**Usage**

To execute the program, use the command:

_bash_

python3 takuzu.py <instance_file>


**Report**

An additional report is available in the "Final Report" folder. This report compares different types of search algorithms to determine the most effective method implemented in the main file, takuzu.py. Additional files were created to support this comparison and can be found in the "additional takuzu modes test" folder.

_Final grade: 19/20_