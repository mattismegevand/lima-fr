import os
import openai
import concurrent.futures

from datasets import load_dataset

from utils import load_jsonl, save_jsonl

NUM_WORKERS = 8
CHECKPOINT_INTERVAL = 100
HEADER = """You are asked to translate the following example task from English into French without providing any explanation.

Here are the requirements:
1. Translate the instruction and the output text if there is one.
2. Ensure faithful translation, and keep the correctness of the example.
3. Maintain the format, keep the "Instruction" and "Output" if they exist in the example.
4. Don't translate the code, including its syntax, and variable names.

"""

def gen_prompt(convs):
  ''' Generate the prompt for OpenAI API. '''
  prompt = HEADER
  prompt += f'Instruction: "{convs[0].strip()}"\n\n'
  if len(convs) > 1: prompt += f'Output: "{convs[1].strip()}"\n'
  return prompt

def gen_messages(prompt):
  ''' Generate the messages for OpenAI API. '''
  return [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt},
  ]

def get_response(messages):
  ''' Get the response from OpenAI API. '''
  return openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=messages,
    temperature=0,
    max_tokens=1024,
  )

def parse_response(response):
  ''' Parse the response from OpenAI API and return the instruction and output. '''
  s = response["choices"][0]["message"]["content"].strip()
  if "\nOutput: " in s:
    inst, out = s.split("\nOutput: ")
    inst, out = inst.strip(), out.strip()
    inst, out = inst[14:-2].strip(), out[1:-1].strip() # remove "Instruction: " and quotes
  else:
    inst = s[14:-1].strip() # remove "Instruction: " and quotes
    out = None
  return inst, out

def translate(item):
  ''' Translate an item from the dataset. '''
  try:
    messages = gen_messages(gen_prompt(item['conversations']))
    response = get_response(messages)
    inst, out = parse_response(response)
    if out:
      return {'conversations': [inst, out], 'source': item['source']}
    else:
      return {'conversations': [inst], 'source': item['source']}
  except Exception as e:
    return {'error': str(e), 'item': item}

def translate_dataset(dataset):
  ''' Translate LIMA dataset to French. '''
  translated_dataset = []
  missed_entries = []

  with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
    futures = {executor.submit(translate, item): item for item in dataset}

    for future in concurrent.futures.as_completed(futures):
      result = future.result()
      if "error" not in result:
        translated_dataset.append(result)
      else:
        missed_entries.append(result)

      print('.', end='', flush=True)

      if len(translated_dataset) % CHECKPOINT_INTERVAL == 0:
        save_jsonl(dataset, "lima-fr_checkpoint.jsonl")

  print() # newline after dots
  return translated_dataset, missed_entries

if __name__ == "__main__":
  if not (os.path.exists("lima_train.jsonl") or os.path.exists("lima_test.jsonl")):
    dataset = load_dataset("GAIR/lima")
    dataset["train"].to_json("lima_train.jsonl")
    dataset["test"].to_json("lima_test.jsonl")

  for split in ["train", "test"]:
    dataset = load_jsonl(f"lima_{split}.jsonl")
    print(f"Loaded {len(dataset)} items from lima_{split}.jsonl")

    print(f"Translating {split} set...")
    translated_dataset, missed_entries = translate_dataset(dataset)

    out_filename = f"lima-fr_{split}.jsonl"
    save_jsonl(translated_dataset, out_filename)
    print(f"Saved {len(translated_dataset)} items to {out_filename}")

    if missed_entries:
      missed_filename = f"missed_entries_{split}.jsonl"
      save_jsonl(missed_entries, missed_filename)
      print(f"Saved {len(missed_entries)} missed entries to {missed_filename}")
