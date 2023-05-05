import logging
from flask import Flask, jsonify, request
from flask_socketio import SocketIO, emit
from wizard import Wizard


def create_model_config():
    return {
        'knn_50': ('knn', {'model__n_neighbors': [50]}),
        'lr': ('lr', {}),
    }


def setup_wizard():
    model_config = create_model_config()
    wzrd = Wizard(
        raw_data_path="backend/data/nba.csv",
        k_best=6,
        model_config=model_config
    )
    display_model_evaluations(wzrd)
    return wzrd


def display_model_evaluations(wzrd):
    for group in wzrd.model_groups:
        wzrd.model_groups[group].display_evaluations_table()


app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")
wzrd = setup_wizard()

logging.basicConfig(
    filename='_log.txt',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


@app.route("/project_player/<name>/<model_name>/<int:n_games>")
def project_player(name, model_name, n_games):
    result = wzrd.project_player(name, model_name, n_games)
    return jsonify(result)


def log_message(level, message):
    colors = {
        0: '\033[1;32m',  # green for success
        1: '\033[1;31m',  # red for error
        2: '\033[1;36m'   # cyan for status
    }
    reset = '\033[0m'  # reset color and formatting

    client_ip = request.environ.get('REMOTE_ADDR')
    client_port = request.environ.get('REMOTE_PORT')
    event = f"[{client_ip}:{client_port}] {message}"
    logging.debug(event)
    print(f"{colors[level]}{event}{reset}")


@socketio.on('connect')
def handle_socket():
    log_message(0, "CONNECTED")


@socketio.on('disconnect')
def handle_disconnect():
    log_message(1, "DISCONNECTED")


@socketio.on('message')
def handle_message(data):
    emit('message', data)
    log_message(2, "MESSAGE")


@socketio.on('status')
def handle_status():
    player_id_team = get_player_id_team()
    status = {
        "active": True,
        "player_id_team": player_id_team
    }
    emit("status", status)
    log_message(2, "STATUS REQUEST")


def get_player_id_team():
    player_id_team = {}
    for team in wzrd.player_team_map:
        for player, pid in wzrd.player_team_map[team]:
            player_id_team[player] = (pid, team)
    return player_id_team


@socketio.on('project_player')
def handle_project_player(data):
    model_name, n_games, name = data["model_name"], data["n_games"], data["name"]
    result = wzrd.project_player(name=name, model_name=model_name, n_games=n_games)
    emit("project_player", result)
    log_message(0, f"{name} | {model_name} | n={n_games}")


@socketio.on('compare_player')
def handle_project_player(data):
    model_name, n_games, name_a, name_b = data["model_name"], data["n_games"], data["name_a"], data["name_b"]
    resultA = wzrd.project_player(name=name_a, model_name=model_name, n_games=n_games)
    resultB = wzrd.project_player(name=name_b, model_name=model_name, n_games=n_games)
    result = {"A": resultA, "B": resultB}
    emit("compare_player", result)
    log_message(0, f"{name_a} v. {name_b} | {model_name} | n={n_games}")


@socketio.on('project_team')
def handle_project_team(data):
    model_name, n_games, team = data["model_name"], data["n_games"], data["team"]
    result = wzrd.project_team_players(team, model_name, n_games)
    emit("project_team", result)
    log_message(0, f"{team} | {model_name} | n={n_games}")


@socketio.on('matchup_team')
def handle_matchup_team(data):
    model_name, n_games, team_a, team_b = data["model_name"], data["n_games"], data["team_a"], data["team_b"]
    resultA = wzrd.project_team_players(team_abbrev=team_a, model_name=model_name, n_games=n_games)
    resultB = wzrd.project_team_players(team_abbrev=team_b, model_name=model_name, n_games=n_games)

    emit("matchup_team", {"A": resultA, "B": resultB})
    log_message(0, f"{team_a} v. {team_b} | {model_name} | n={n_games}")


@app.route('/')
def initial_data():
    log_message(2, "MANIFEST")
    return jsonify({
        'initial_time': wzrd.start_time,
        'model_config_list': list(create_model_config().keys()),
        'player_team_map': wzrd.player_team_map
    })


if __name__ == '__main__':
    socketio.run(app, host='127.0.0.1', port=5000)
