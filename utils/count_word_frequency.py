import pandas as pd
import time

def count_word_frequency(database, output_dir='output', remove_single_char=True):

    subset = database['cleaned_text']
    text = subset.values.tolist()

    # Create an empty dictionary
    d = dict()
    
    # Loop through each line of the file
    for line in text:
        # Remove the leading spaces and newline character
        line = line.strip()
    
        # Convert the characters in line to
        # lowercase to avoid case mismatch
        line = line.lower()
        # Split the line into words
        words = line.split(" ")
                            
        # Iterate over each word in line
        for word in words:
            if word == "":
                continue
            if remove_single_char and len(word) == 1:
                continue
            # Check if the word is already in dictionary
            if word in d:
                # Increment count of word by 1
                d[word] = d[word] + 1
            else:
                # Add the word to dictionary with count 1
                d[word] = 1

    # sort the dictionary by the frequency count values, in descender order
    sortedDict = {k: v for k, v in sorted(d.items(), key=lambda item: item[1], reverse=True)}

    current_time = time.strftime("%Y%m%d-%H%M%S")
    rankingFile = f"{output_dir}/word_ranking_{current_time}.txt"
    # make a new text file, then write results to that
    with open(rankingFile, 'w', encoding='utf-8') as f:
        for key in list(sortedDict.keys()):
            f.write(f"{key}: {sortedDict[key]}\n")
            print(f"{key}: {sortedDict[key]}")