import requests

player_name = "LeBron James"
model_name = "knn_10"
n_games = 3
url = f"http://localhost:5000/project-player/{player_name}/{model_name}/{n_games}"

response = requests.get(url)
print(f"Raw response text: {response.text}")  # Add this line
data = response.json()

print(f"Status Code: {data['status_code']}")
print("Data:")
print(data['data'])
