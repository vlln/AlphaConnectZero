## AlphaZero for Connect Game

This is a PyTorch implementation of AlphaGoZero for Connect Game.

**ATTENTION: This is a work in progress. The code is not complete and may not work properly.**

### Connect Game
Connect Game is a two-player game. The game board is a `nxn` grid. Players take turns placing a piece on the board. The first player to align `k` pieces horizontally, vertically, or diagonally wins the game.

Here are some specifications of the game:
- Gomoku: `n=15, k=5`. 
- Tic-Tac-Toe: `n=3, k=3`.
- Connect Four: `n=8, k=4`. Only one colum can be selected at a time.

### Requirements
- numpy>=1.24.4
- torch>=2.4.1
- loguru>=0.7.2
[ ] add a self-play noise
[ ] inheriting the tree during self-play
[ ] random select a oponent during evaluation
### Getting Started




