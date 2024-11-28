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
from re import compile


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

    def __repr__(self):
        return f'Token(ttype={self.ttype}, value=\'{self.value}\')'


# Scan input file lines to extract tokens
NUMBERS_PATTERN = compile(r'(-)?\d+([.]\d+((e|E)(-|\+)?\d+)?)?')
read_tokens: list[Token] = []

for line in open(argv[1], 'r').readlines():
    i = 0
    while i < len(line):
        if line[i] != ' ':
            if m := NUMBERS_PATTERN.match(line, i):
                read_tokens.append(Token(TokenType.NUMBER, m.group()))
            elif line[i] == '+':
                read_tokens.append(Token(TokenType.PLUS, '+'))
            elif line[i] == '-':
                read_tokens.append(Token(TokenType.MINUS, '-'))
            elif line[i] == '*':
                read_tokens.append(Token(TokenType.STAR, '*'))
            elif line[i] == '/':
                read_tokens.append(Token(TokenType.SLASH, '/'))
            i += len(read_tokens[-1].value)
        else:
            i += 1

print(read_tokens)
