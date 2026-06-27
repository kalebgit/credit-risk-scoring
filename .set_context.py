import os
import json
from pathlib import Path

path = Path.home() / ".vim" / "coc-settings.json"
new_path = Path(__file__).parent / ".venv" / "lib" / "python"

with open(path) as f: 
    data = json.load(f)


data["python.pythonPath"] = str(new_path)

with open(path, "w") as f: 
    json.dump(data, f, indent="\t", ensure_ascii=False)
