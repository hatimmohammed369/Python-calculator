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


# Context-Free Grammar
# Expression => Term ( ( '+' | '-' )? Term )?
# Term => Factor ( ( '*' | '/' ) Factor )?
# Factor => ( '-' )? Exponential
# Exponential => Atomic ( '**' Atomic )?
# Atomic => NUMBER | '(' Expression ')'


class ExpressionType(Enum):
    TOP_CLASS = 1
    TERM = 2
    FACTOR = 3
    EXPONENTIAL = 4
    ATOMIC = 5


class ExpressionBase:
    def __init__(self, expression_type: ExpressionType):
        self.expression_type: ExpressionType = expression_type


class Expression(ExpressionBase):
    def __init__(self, lhs, op, rhs):
        super().__init__(ExpressionType.TOP_CLASS)
        self.left_term: ExpressionBase = lhs
        self.operator: Token = op
        self.right_term: ExpressionBase = rhs


class Term(ExpressionBase):
    def __init__(self, lhs, op, rhs):
        super().__init__(ExpressionType.TERM)
        self.left_factor: ExpressionBase = lhs
        self.operator: Token = op
        self.right_factor: ExpressionBase = rhs


class Factor(ExpressionBase):
    def __init__(self, negated: bool, exponential):
        super().__init__(ExpressionType.FACTOR)
        self.negated: bool = negated
        self.exponential: ExpressionBase = exponential


class Exponential(ExpressionBase):
    def __init__(self, base, exponent):
        super().__init__(ExpressionType.EXPONENTIAL)
        self.base: ExpressionBase = base
        self.exponent: ExpressionBase = exponent


class Atomic(ExpressionBase):
    def __init__(self, value):
        super().__init__(ExpressionType.ATOMIC)
        self.is_number = (value is Token)
        self.value: Token | ExpressionBase = value
