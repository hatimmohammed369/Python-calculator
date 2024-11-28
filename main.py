# How to run:
# python main.py path_to_input_file

# 1 - Implement basic arithmetic
# 2 - Implement nested expressions
# 3 - Implement multi-lined processing
# (\n to end expression, \ to continue expression in next line)
# 4 - Implement variables
# 5 - Implement function calls

# Tokens:
# Numbers
# + - * /

import sys
from sys import argv

input_file = open(argv[0], 'r')

# Print input file
for line in input_file.readlines():
    print(line, end='')
print()
