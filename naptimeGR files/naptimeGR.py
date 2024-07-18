#!/usr/bin/env python
import chess
import random
import sys
import numpy as np
import pickle


def extract_features_from_fen(fen):
    """
    Extracts features from the given FEN string.
    """
    piece_values = {'P': 1, 'N': 3, 'B': 3, 'R': 5, 'Q': 9, 'K': 0,  # Piece values
                    'p': -1, 'n': -3, 'b': -3, 'r': -5, 'q': -9, 'k': 0}

    # Initialize feature vector
    features = np.zeros(82)  # Change the size to 82

    # Split FEN string into parts
    fen_parts = fen.split()

    # Extract piece positions from FEN
    piece_placement = fen_parts[0]
    rank_index = 0
    file_index = 0
    for char in piece_placement:
        if char == '/':
            rank_index += 1
            file_index = 0
        elif char.isdigit():
            file_index += int(char)
        else:
            # Convert piece character to piece value
            piece_value = piece_values[char]

            # Convert rank and file indices to square index
            square_index = rank_index * 8 + file_index

            # Store piece value in feature vector
            features[square_index] = piece_value

            file_index += 1

    # Add additional features
    if fen_parts[1] == 'w':
        features[64] = 1
    elif fen_parts[1] == 'b':
        features[64] = -1

    return features.reshape(1, -1)


class Naptime:
    def __init__(self, model_file):
        self.board = chess.Board()
        self.model = self.load_model(model_file)

    def load_model(self, model_file):
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
        return model

    def make_move(self):
        legal_moves = list(self.board.legal_moves)
        if legal_moves:
            # Initialize variables to store the best move and its evaluation
            best_move = None
            best_evaluation = -float('inf')

            # Iterate through legal moves and evaluate each move
            for move in legal_moves:
                self.board.push(move)
                evaluation = self.evaluate_position()
                self.board.pop()
                if evaluation > best_evaluation:
                    best_evaluation = evaluation
                    best_move = move
            return best_move
        else:
            print("stalemate")
            sys.exit(0)

    def evaluate_position(self):
        # Extract features from the current board position
        features = extract_features_from_fen(self.board.fen())

        # Use the model to predict the evaluation
        evaluation = self.model.predict(features)[0]

        return evaluation

    def uci(self, msg: str):
        if msg == "uci":
            print("id name NaptimeGR")
            print("id Ivan Kwetey")
            print("uciok")
        elif msg == "isready":
            print("readyok")
        elif msg.startswith("position startpos moves"):
            self.board.clear()
            self.board.set_fen(chess.STARTING_FEN)
            moves = msg.split()[3:]
            for move in moves:
                self.board.push(chess.Move.from_uci(move))
        elif msg.startswith("position fen"):
            fen = msg.removeprefix("position fen ")
            self.board.set_fen(fen)
        elif msg.startswith("go"):
            move = self.make_move()
            print(f"bestmove {move}")
        elif msg == "quit":
            sys.exit(0)

    def run(self):
        try:
            while True:
                self.uci(input())
        except Exception as e:
            print(f"Fatal Error: {e}")


if __name__ == "__main__":
    model_file = "naptime_GradientBoostingRegressor.pkl"  # Path to your trained model file
    bot = Naptime(model_file)
    bot.run()
