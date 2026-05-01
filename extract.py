import json

with open('speech-emotion-updated-transformer (1).ipynb', 'r', encoding='utf-8') as f:
    data = json.load(f)

with open('notebook_code.py', 'w', encoding='utf-8') as f:
    for cell in data['cells']:
        if cell['cell_type'] == 'code':
            f.write("".join(cell['source']) + "\n\n")
