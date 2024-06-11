#!/bin/bash

# Define the size threshold (in bytes). 0.2 MB = 204800 bytes.
size_threshold=204800

# Create a new gitignore file
echo "# Ignore files larger than 0.2MB" > .gitignore

# Find files larger than the size threshold and append to .gitignore
find . -type f -size +${size_threshold}c >> .gitignore

# Remove duplicates from .gitignore
sort -u -o .gitignore .gitignore

# Remove old git directory and initialize a new one
rm -rf .git
git init

# Add and commit files except those in .gitignore
git add .
git commit -m "Initial commit excluding large files"

# Add remote and push
git remote add origin https://github.com/bdytx5/grokking.git
git push -u origin main

