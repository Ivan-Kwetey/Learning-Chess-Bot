import csv
from chester.timecontrol import TimeControl
from chester.tournament import play_tournament

# Each string is the name/path to an executable UCI engine.
players = ["./random_chess_bot.py", "./naptimeGR.py"]

# Specify time and increment, both in seconds.
time_control = TimeControl(initial_time=2, increment=0)

# Play each match-up twice.
n_games = 20

# Tabulate scores at the end.
scores = {}

# List to store game results
game_results = []

# Function to extract player names without './' prefix
def extract_player_name(player_path):
    return player_path.split("/")[-1].split(".")[0]

for i, pgn in enumerate(play_tournament(
    players,
    time_control,
    n_games=n_games,
    repeat=True,  # Each opening played twice,
)):
    # Printing out the game result.
    pgn.headers["Event"] = "CS5100 Tournament"
    pgn.headers["Site"] = "My Computer"
    print(pgn, "\n")

    # Update scores.
    white = pgn.headers["White"]
    black = pgn.headers["Black"]
    result = pgn.headers["Result"]

    if result == "1-0":  # White wins
        scores.setdefault(white, {"Wins": 0, "Losses": 0, "Draws": 0})
        scores.setdefault(black, {"Wins": 0, "Losses": 0, "Draws": 0})
        scores[white]["Wins"] += 1
        scores[black]["Losses"] += 1
        game_results.append({
            "Round": i // len(players) + 1,
            "White": extract_player_name(white),
            "Black": extract_player_name(black),
            extract_player_name(players[0]): 1,
            extract_player_name(players[1]): 0,
            "Site": "My Computer"
        })
    elif result == "0-1":  # Black wins
        scores.setdefault(white, {"Wins": 0, "Losses": 0, "Draws": 0})
        scores.setdefault(black, {"Wins": 0, "Losses": 0, "Draws": 0})
        scores[black]["Wins"] += 1
        scores[white]["Losses"] += 1
        game_results.append({
            "Round": i // len(players) + 1,
            "White": extract_player_name(white),
            "Black": extract_player_name(black),
            extract_player_name(players[0]): 0,
            extract_player_name(players[1]): 1,
            "Site": "My Computer"
        })
    elif result == "1/2-1/2":  # Draw
        scores.setdefault(white, {"Wins": 0, "Losses": 0, "Draws": 0})
        scores.setdefault(black, {"Wins": 0, "Losses": 0, "Draws": 0})
        scores[white]["Draws"] += 1
        scores[black]["Draws"] += 1
        game_results.append({
            "Round": i // len(players) + 1,
            "White": extract_player_name(white),
            "Black": extract_player_name(black),
            extract_player_name(players[0]): 0,
            extract_player_name(players[1]): 0,
            "Site": "My Computer"
        })
    else:
        print("Invalid result:", result)

# Write game results to CSV file
csv_filename = 'tournament_results.csv'
with open(csv_filename, 'w', newline='') as csvfile:
    fieldnames = ["Round", "White", "Black", extract_player_name(players[0]), extract_player_name(players[1]), "Site"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(game_results)

# Print player scores
print("\nPlayer Scores:")
for player, score in scores.items():
    print(f"{player}: Wins - {score['Wins']}, Losses - {score['Losses']}, Draws - {score['Draws']}")

print(f"\nTournament results saved to {csv_filename}")
