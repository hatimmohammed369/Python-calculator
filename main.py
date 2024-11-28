# How to run:
# python main.py path_to_input_file

# Tokens:
# Numbers
# + - * /

import sys

input_file = open(sys.argv[0], 'r')

# Print input file
for line in input_file.readlines():
    print(line, end='')
print()
