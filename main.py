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

from enum import Enum
from sys import argv

input_file = open(argv[0], 'r')

# Print input file
for line in input_file.readlines():
    print(line, end='')
print()


class TokenType(Enum):
    NUMBER = 1
    PLUS = 2
    MINUS = 3
    STAR = 4
    SLASH = 5


class Token:
    # Constructor
    def __init__(self, ttype: TokenType, value: str):
        self.ttype = ttype
        self.value = value


