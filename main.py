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
from sys import argv, stderr
from re import compile
from typing import override


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


# Context-Free Grammar
# Expression => Term ( ( '+' | '-' ) Term )?
# Term => Exponential ( ( '*' | '/' ) Exponential )?
# Exponential => Atomic ( '**' Atomic )?
# Atomic => ( '-' )? ( NUMBER | '(' Expression ')' )


class ExpressionType(Enum):
    TOP_CLASS = 1
    TERM = 2
    EXPONENTIAL = 3
    ATOMIC = 4


class ExpressionBase:
    def __init__(self, expression_type: ExpressionType):
        self.expression_type: ExpressionType = expression_type

    def evaluate(self) -> float:
        pass

    def __repr__(self) -> str:
        pass


class Expression(ExpressionBase):
    def __init__(self, lhs, op, rhs):
        super().__init__(ExpressionType.TOP_CLASS)
        self.left_term: ExpressionBase = lhs
        self.operator: Token = op
        self.right_term: ExpressionBase = rhs

    # Expression => Term ( ( '+' | '-' ) Term )?
    @override
    def evaluate(self) -> float:
        left = self.left_term.evaluate()
        right = self.right_term.evaluate()
        if self.operator.value == '+':
            return left + right
        return left - right

    @override
    def __repr__(self) -> str:
        return f'{self.left_term} {self.operator.value} {self.right_term}'


class Term(ExpressionBase):
    def __init__(self, lhs, op, rhs):
        super().__init__(ExpressionType.TERM)
        self.left_exponential: ExpressionBase = lhs
        self.operator: Token = op
        self.right_exponential: ExpressionBase = rhs

    # Term => Exponential ( ( '*' | '/' ) Exponential )?
    @override
    def evaluate(self) -> float:
        left = self.left_exponential.evaluate()
        right = self.right_exponential.evaluate()
        if self.operator.value == '*':
            return left * right
        return left / right

    @override
    def __repr__(self) -> str:
        return f'{self.left_exponential} \
            {self.operator.value} {self.right_exponential}'


class Exponential(ExpressionBase):
    def __init__(self, base, exponent):
        super().__init__(ExpressionType.EXPONENTIAL)
        self.base: ExpressionBase = base
        self.exponent: ExpressionBase = exponent

    # Exponential => Atomic ( '**' Atomic )?
    @override
    def evaluate(self) -> float:
        base = self.base.evaluate()
        exponent = self.exponent.evaluate()
        return base ** exponent

    @override
    def __repr__(self) -> str:
        return f'{self.base} ** {self.exponent}'


class Atomic(ExpressionBase):
    def __init__(self, is_negated, is_number, value):
        super().__init__(ExpressionType.ATOMIC)
        self.is_negated = is_negated
        self.is_number = is_number
        self.value: Token | ExpressionBase = value

    # Atomic => ( '-' )? ( NUMBER | '(' Expression ')' )
    @override
    def evaluate(self) -> float:
        sign = 1
        if self.is_negated:
            sign = -1
        if self.is_number:
            return sign * float(self.value.value)
        return sign * self.value.evaluate()

    @override
    def __repr__(self) -> str:
        sign = ''
        if self.is_negated:
            sign = '-'
        value = ''
        if self.is_number:
            value = self.value.value
        else:
            value = f'({self.value})'
        return f'{sign}{value}'


class Parser:
    def __init__(self):
        self.tokenizer = Tokenizer()
        self.read_next_token()

    # Expression => Term ( ( '+' | '-' ) Term )?
    def parse_expression(self) -> ExpressionBase:
        if left_term := self.parse_term():
            if self.check(TokenType.PLUS) or self.check(TokenType.MINUS):
                operator = self.consume()
                if right_term := self.parse_term():
                    return Expression(
                        lhs=left_term,
                        op=operator,
                        rhs=right_term
                    )
                else:
                    print('Expected expression after ' +
                          operator.value, file=stderr)
            else:
                return left_term
        return None

    # Term => Exponential ( ( '*' | '/' ) Exponential )?
    def parse_term(self) -> ExpressionBase:
        if left_exponential := self.parse_exponential():
            if self.check(TokenType.STAR) or self.check(TokenType.SLASH):
                operator = self.consume()
                if right_exponential := self.parse_exponential():
                    return Term(
                        lhs=left_exponential,
                        op=operator,
                        rhs=right_exponential
                    )
                else:
                    print('Expected expression after ' +
                          operator.value, file=stderr)
            else:
                return left_exponential
        return None

    # Exponential => Atomic ( '**' Atomic )?
    def parse_exponential(self) -> ExpressionBase:
        if base := self.parse_atomic():
            if self.check(TokenType.EXPONENT):
                self.read_next_token()
                if exponent := self.parse_atomic():
                    return Exponential(base, exponent)
                else:
                    print('Expected expression after **', file=stderr)
            else:
                return base
        return None

    # Atomic => ( '-' )? ( NUMBER | '(' Expression ')' )
    def parse_atomic(self) -> ExpressionBase:
        is_negated = self.check(TokenType.MINUS)
        if is_negated:
            self.read_next_token()
        if self.check(TokenType.NUMBER):
            if value := self.consume():
                return Atomic(is_negated, True, value)
            else:
                print('Expected expression after -', file=stderr)
        elif self.check(TokenType.LEFT_PARENTHESIS):
            self.read_next_token()
            if value := self.parse_expression():
                if self.check(TokenType.RIGHT_PARENTHESIS):
                    self.read_next_token()
                    return Atomic(is_negated, False, value)
                else:
                    print('Expected ) after expression', file=stderr)
            else:
                print('Expected expression after (', file=stderr)
        return None

    def consume(self):
        copy = self.current
        self.read_next_token()
        return copy

    def check(self, ttype: TokenType) -> bool:
        return self.current and self.current.ttype == ttype

    def read_next_token(self):
        try:
            self.current: Token = next(self.tokenizer)
        except StopIteration:
            self.current: Token = None


print(Parser().parse_expression().evaluate())
