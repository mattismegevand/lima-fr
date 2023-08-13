import json

def load_jsonl(filename):
  ''' Load a JSONL file into a list of dictionaries. '''
  data = []
  with open(filename) as f:
    for line in f:
      data.append(json.loads(line))
  return data

def save_jsonl(data, filename):
  ''' Save a list of dictionaries to a JSONL file. '''
  with open(filename, "w") as f:
    for item in data:
      f.write(json.dumps(item) + "\n")
