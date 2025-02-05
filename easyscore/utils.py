import re
import json
from datetime import datetime
from pathlib import Path

def find_json_blocks(input_string):
    """
    Extracts and parses JSON code blocks from a given input string.
    This function searches for JSON code blocks enclosed within triple backticks
    (```json ... ```) in the input string, parses each block, and returns a list
    of dictionaries representing the JSON objects.
    Args:
        input_string (str): The input string containing JSON code blocks.
    Returns:
        list: A list of dictionaries parsed from the JSON code blocks found in the input string.
    """
    
    # Regular expression to find JSON code blocks
    json_blocks = re.findall(r'```json(.*?)```', input_string, re.DOTALL)
    
    # Parse each JSON block and store in a list
    json_dicts = []
    for block in json_blocks:
        try:
            json_dict = json.loads(block.strip())
            json_dicts.append(json_dict)
        except json.JSONDecodeError:
            print("Invalid JSON block found and skipped.")
    return json_dicts



def get_fname():
    fp = f"./runs/{datetime.now().strftime("%Y%m%d%H%M%S")}.json"
    Path(fp).mkdir(parents=True, exist_ok=True)
    return fp
