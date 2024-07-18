import csv
import pickle
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import chess


def clean_fen(fen):
    # Define cleaning rules here
    cleaned_fen = fen.replace('/', '')  # Remove forward slashes

    return cleaned_fen


def fen_to_features(fen):
    """
    Converts a cleaned FEN string to a feature vector suitable for machine learning.
    """
    feature_vector = []

    # Board state features
    board_state = fen.split(' ')[0]  # Extract board state part from FEN

    # Count of each piece type for white and black
    white_piece_count = {'P': 0, 'N': 0, 'B': 0, 'R': 0, 'Q': 0, 'K': 0}
    black_piece_count = {'p': 0, 'n': 0, 'b': 0, 'r': 0, 'q': 0, 'k': 0}

    for char in board_state:
        if char.isdigit():
            feature_vector.extend([0] * int(char))  # Empty squares
        elif char.islower():
            feature_vector.append(-1)  # Black pieces
            black_piece_count[char] += 1
        else:
            feature_vector.append(1)  # White pieces
            white_piece_count[char.upper()] += 1

    # Add piece count features
    for piece in ['P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k']:
        white_count = white_piece_count.get(piece, 0)
        black_count = black_piece_count.get(piece.lower(), 0)
        feature_vector.append(white_count - black_count)

    # Material balance
    white_material = 1 * white_piece_count['P'] + 3 * white_piece_count['N'] + \
                     3 * white_piece_count['B'] + 5 * white_piece_count['R'] + \
                     9 * white_piece_count['Q']
    black_material = 1 * black_piece_count['p'] + 3 * black_piece_count['n'] + \
                     3 * black_piece_count['b'] + 5 * black_piece_count['r'] + \
                     9 * black_piece_count['q']
    feature_vector.append(white_material - black_material)

    # Other FEN features
    fen_features = fen.split(' ')[1:]
    for feature in fen_features:
        if feature == 'w':
            feature_vector.append(1)  # White to move
        elif feature == 'b':
            feature_vector.append(-1)  # Black to move
        elif feature in ('K', 'Q', 'k', 'q'):
            feature_vector.append(1)  # Castling available
        else:
            feature_vector.append(0)  # No relevant feature

    return feature_vector


def load_dataset(filename):
    """
    Loads the dataset from a CSV file.
    """
    X = []  # Features
    y = []  # Targets
    with open(filename, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            fen = row['fen']
            cleaned_fen = clean_fen(fen)
            if cleaned_fen:
                fen_features = fen_to_features(cleaned_fen)
                X.append(fen_features)
                y.append(float(row['eval']))
            else:
                print(f"Ignoring invalid FEN: {fen}")
    return np.array(X), np.array(y)


def augment_data(X, y):
    """
    Augments the dataset by applying transformations to the positions.
    """
    augmented_X = []
    augmented_y = []
    for fen_features, eval_score in zip(X, y):
        # Flip the board horizontally to generate a new position
        flipped_features = fen_features[::-1]
        augmented_X.append(flipped_features)
        augmented_y.append(eval_score)

        # Rotate the board 90 degrees clockwise
        rotated_features = np.rot90(np.array(fen_features).reshape(1, 82), 1).flatten()
        augmented_X.append(rotated_features.tolist())
        augmented_y.append(eval_score)

        # Rotate the board 180 degrees clockwise
        rotated_features = np.rot90(np.array(fen_features).reshape(1, 82), 2).flatten()
        augmented_X.append(rotated_features.tolist())
        augmented_y.append(eval_score)

    return np.array(augmented_X), np.array(augmented_y)


def train_model(X_train, y_train, model_save_path):
    """
    Trains a Gradient Boosting Regressor model using the provided training data and saves it.
    """
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.3]
    }
    model = GradientBoostingRegressor()
    grid_search = GridSearchCV(model, param_grid, cv=3, verbose=2, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    # Save the trained model using pickle
    with open(model_save_path, 'wb') as model_file:
        pickle.dump(best_model, model_file)
    print(f"Trained model saved at: {model_save_path}")
    return best_model


def evaluate_model(model, X_test, y_test, csv_filename):
    """
    Evaluates the trained model on the test data and prints the mean squared error,
    mean absolute error, root mean squared error, and R-squared.
    """
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # Print the metrics
    print(f"Mean Absolute Error: {mae}")
    print(f"Mean Squared Error: {mse}")
    print(f"Root Mean Squared Error: {rmse}")
    print(f"R-squared: {r2}")

    # Save the metrics to a CSV file
    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = ['Metric', 'Value']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({'Metric': 'Mean Absolute Error', 'Value': mae})
        writer.writerow({'Metric': 'Mean Squared Error', 'Value': mse})
        writer.writerow({'Metric': 'Root Mean Squared Error', 'Value': rmse})
        writer.writerow({'Metric': 'R-squared', 'Value': r2})

    print(f"Metrics saved to {csv_filename}")


def evaluate_position_with_model(fen, model):
    """
    Evaluates a position using the learned model.
    """
    features = fen_to_features(clean_fen(fen))
    evaluation = model.predict([features])
    return evaluation[0]


def main():
    # Load dataset
    X, y = load_dataset('chess_evaluations4.csv')

    # Augment data
    X_augmented, y_augmented = augment_data(X, y)

    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_augmented, y_augmented, test_size=0.2, random_state=42)

    # Train model and save it
    model_save_path = 'chess_model_GradientBoostingRegressor.pkl'
    model = train_model(X_train, y_train, model_save_path)

    # Evaluate model
    csv_filename = 'GradientBoostingRegressor_metrics.csv'
    evaluate_model(model, X_test, y_test, csv_filename)

    # Test the model with a FEN string
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    evaluation = evaluate_position_with_model(fen, model)
    print(f"Evaluation for FEN '{fen}': {evaluation}")


if __name__ == "__main__":
    main()
