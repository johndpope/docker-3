import json

def load_jsonl(filename):
    d_list = []
    with open(filename, encoding='utf-8', mode='r') as in_f:
        for line in in_f:
            item = json.loads(line.strip())
            d_list.append(item)

    return d_list 