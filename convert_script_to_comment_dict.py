import re
import json

def parse_comments_file(file_path):
    """
    Parses a text file into a dictionary for the chess animator.
    Format: [KEY] followed by the comment text.
    """
    with open(file_path, 'r') as f:
        content = f.read()

    # Regex to find [KEY] and the text following it until the next [KEY]
    pattern = r'\[(.*?)\]\s*(.*?)(?=\s*\[|$)'
    matches = re.findall(pattern, content, re.DOTALL)

    comments_dict = {}
    for key, text in matches:
        # Clean up the key and the text
        clean_key = key.strip().lower()
        clean_text = " ".join(text.split()) # Removes newlines/extra spaces
        
        # Store in dictionary
        comments_dict[clean_key] = clean_text

    return comments_dict

# Example Usage:
# COMMENTS = parse_comments_file("game_notes.txt")
