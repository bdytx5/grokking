#!/bin/bash

# Define the size threshold (in bytes). 0.2 MB = 204800 bytes.
size_threshold=204800

# Find files larger than the size threshold and append to .gitignore
find . -type f -size +${size_threshold}c -exec echo {} \; >> .gitignore

# Remove any duplicates in the .gitignore file
sort -u -o .gitignore .gitignore

echo ".gitignore updated with files larger than 0.2 MB"

