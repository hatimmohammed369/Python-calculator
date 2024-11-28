# How to run:
# python main.py path_to_input_file

# 1 - Implement basic arithmetic
# + - * / ** -(Expression)
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
    EXPONENT = 6
    LEFT_PARENTHESIS = 7
    RIGHT_PARENTHESIS = 8


class Token:
    # Constructor
    def __init__(self, ttype: TokenType, value: str):
        self.ttype: TokenType = ttype
        self.value: str = value

    def __repr__(self):
        return f'Token(ttype={self.ttype}, value=\'{self.value}\')'


# Scan input file lines to extract tokens
NUMBERS_PATTERN = compile(r'(-)?\d+([.]\d+((e|E)(-|\+)?\d+)?)?')


class Tokenizer:
    def __init__(self):
        self.input_lines: list[str] = open(argv[1], 'r').readlines()
        self.line: int = 0
        self.pos: int = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.line < len(self.input_lines):
            current_line = self.input_lines[self.line]

            if self.pos >= len(current_line):
                self.pos = 0
                self.line += 1
                current_line = self.input_lines[self.line]

            while self.pos < len(current_line):
                if current_line[self.pos] != ' ':
                    read_token = None
                    if m := NUMBERS_PATTERN.match(current_line, self.pos):
                        read_token = Token(TokenType.NUMBER, m.group())
                    elif current_line[self.pos] == '+':
                        read_token = Token(TokenType.PLUS, '+')
                    elif current_line[self.pos] == '-':
                        read_token = Token(TokenType.MINUS, '-')
                    elif current_line[self.pos] == '*':
                        if self.pos+1 < len(current_line) and \
                                current_line[self.pos+1] == '*':
                            read_token = Token(TokenType.EXPONENT, '**')
                        else:
                            read_token = Token(TokenType.STAR, '*')
                    elif current_line[self.pos] == '/':
                        read_token = Token(TokenType.SLASH, '/')
                    elif current_line[self.pos] == '(':
                        read_token = Token(TokenType.LEFT_PARENTHESIS, '(')
                    elif current_line[self.pos] == ')':
                        read_token = Token(TokenType.RIGHT_PARENTHESIS, ')')

                    if read_token:
                        self.pos += len(read_token.value)
                        return read_token
                    else:
                        break
                else:
                    self.pos += 1
        raise StopIteration


print(list(Tokenizer()))
