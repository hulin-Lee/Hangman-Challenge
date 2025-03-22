import json
import os
from openai import OpenAI
from tqdm import tqdm
import re

client = OpenAI(api_key="sk-134004841bed4117a27077bb8106cc78", base_url="https://api.deepseek.com")  # DeepSeek-v3

model_name = 'deepseek-chat'

def predict(masked_word, excluded_letters, q=None, a=None):
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": """
                You are an expert in the Hangman Challenge. Your task is to predict the most probable missing letter in a given masked word, where missing letters are represented by '_'. 

                **Constraints:**
                1. You are given a list of **excluded letters** that should NOT be considered in your prediction.
                2. Letters that **already appear in the masked word** should also be excluded from your guess.

                **Response Format:**  
                Your answer must strictly follow this format:  
                `"The most probable letter is: [letter]"`  

                **Example:**  
                - Input:  
                  - Masked word: `_pp_e`  
                  - Excluded letters: `o, n`  
                - Output: `"The most probable letter is: a"`

                Ensure that your prediction **does not include any excluded or already present letters**.
                """
            },
            {
                "role": "user",
                "content": f"The masked word is: {masked_word}. The excluded letters are: {', '.join(excluded_letters)}."
            }
        ],
        temperature=0.0,
        stream=False
    )
    return response.choices[0].message.content


def filter_string(s):
    return re.sub(r"[^a-zA-Z:_]", "", s)
def find_colon_index(s):
    matches = list(re.finditer(r"mostprobableletteris:", s))
    
    if matches:
        return matches[-1].end() - 1  # return the index of `:` from the last match
    
    return -1  # return -1 if no match is found
def find_ans(ans):
    splited = ans.split()
    new_splited = []
    for s in splited:
        new_splited.append(filter_string(s))
    new_ans = ''.join(new_splited)
    idx = find_colon_index(new_ans)
    if idx == -1:
        return '?'
    else:
        return new_ans[idx+1]
def hangman_challenge(word):
    n = len(word)
    masked_word_list = ['_'] * n
    excluded_letters = []
    wrong_time = 0
    while wrong_time < 6:
        masked_word = ''.join(masked_word_list)
        ans = predict(masked_word, excluded_letters)
        guss = find_ans(ans)
        guss = guss.lower()
        if guss == '?' or guss in excluded_letters:
            print(masked_word_list, excluded_letters, ans)
            return False
        # print(masked_word_list, excluded_letters, ans, guss)
        if guss in word:
            for i in range(n):
                if word[i] == guss:
                    masked_word_list[i] = guss
            if '_' not in masked_word_list:
                return True
        else:
            wrong_time += 1
            excluded_letters.append(guss)
    return False
    


# Read input JSONL file and process each entry
input_file = f'validation.jsonl'
output_file = f'ds_performance.jsonl'
checkpoint_file = f'ds_checkpoint.txt'

# Function to get the last processed line number
def get_last_processed_line():
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            return int(f.read().strip())
    return 0

# Function to update the checkpoint
def update_checkpoint(line_number):
    with open(checkpoint_file, 'w') as f:
        f.write(str(line_number))

# Function to count total lines in the input file
def count_lines(filename):
    with open(filename, 'r') as f:
        return sum(1 for _ in f)

last_processed_line = get_last_processed_line()
total_lines = count_lines(input_file)

try:
    with open(input_file, 'r') as infile, open(output_file, 'a') as outfile:
        # Create a progress bar
        pbar = tqdm(total=total_lines, initial=last_processed_line, desc="Processing", unit="line")
        
        for i, line in enumerate(infile):
            if i < last_processed_line:
                continue  # Skip already processed lines
            
            try:
                entry = json.loads(line)
                word = entry['word']
                state = hangman_challenge(word)
                entry['state'] = state
                
                # write to file
                json.dump(entry, outfile)
                outfile.write('\n')
                
                # Update the checkpoint after each successful processing
                update_checkpoint(i + 1)
                
                # Update the progress bar
                pbar.update(1)
                
                # Calculate and display percentage completion
                percent_complete = ((i + 1) / total_lines) * 100
                pbar.set_postfix({"Percent Complete": f"{percent_complete:.2f}%"})
            
            except Exception as e:
                print(f"\nAn error occurred while processing line {i + 1}: {str(e)}")
                quit()

        pbar.close()

    print(f"\nProcessing complete. Results saved to {output_file}")

except KeyboardInterrupt:
    print("\nProcessing interrupted. Progress saved. You can resume later.")

finally:
    # Ensure the checkpoint is updated even if an error occurs
    update_checkpoint(i)
    
