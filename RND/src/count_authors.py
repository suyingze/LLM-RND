import json

file_path = "../dataset/valid/whole_author_profiles.json"

with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

author_count = len(data)

print("作者数量:", author_count)