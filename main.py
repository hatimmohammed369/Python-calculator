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
from abc import ABC, abstractmethod


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
# Expression => Term ( ( '+' | '-' ) Term )*
# Term => Exponential ( ( '*' | '/' ) Exponential )*
# Exponential => Atomic ( '**' Atomic )?
# Atomic => NUMBER | ( '-' )? ( Atomic | '(' Expression ')' )


class ExpressionBase(ABC):
    @abstractmethod
    def evaluate(self) -> float:
        pass

    @abstractmethod
    def __repr__(self) -> str:
        pass


class Expression(ExpressionBase):
    def __init__(self, lhs, op, rhs):
        self.left_term: ExpressionBase = lhs
        self.operator: Token = op
        self.right_term: ExpressionBase = rhs

    # Expression => Term ( ( '+' | '-' ) Term )*
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
        self.left_exponential: ExpressionBase = lhs
        self.operator: Token = op
        self.right_exponential: ExpressionBase = rhs

    # Term => Exponential ( ( '*' | '/' ) Exponential )*
    @override
    def evaluate(self) -> float:
        left = self.left_exponential.evaluate()
        right = self.right_exponential.evaluate()
        if self.operator.value == '*':
            return left * right
        return left / right

    @override
    def __repr__(self) -> str:
        value = f'{self.left_exponential} '
        value += f'{self.operator.value} '
        value += f'{self.right_exponential}'
        return value


class Exponential(ExpressionBase):
    def __init__(self, base, exponent):
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
    def __init__(self, is_signed, is_number, value):
        self.is_signed = is_signed
        self.is_number = is_number
        self.value: Token | ExpressionBase = value

    # Atomic => NUMBER | ( '-' )? ( Atomic | '(' Expression ')' )
    @override
    def evaluate(self) -> float:
        if self.is_number:
            value = float(self.value.value)
        else:
            value = self.value.evaluate()
            if self.is_signed:
                value *= -1
        return value

    @override
    def __repr__(self) -> str:
        sign = ''
        if self.is_signed:
            sign = '-'
        value = ''
        if self.is_number:
            value = self.value.value
        else:
            value = f'({self.value})'
        return f'{sign}{value}'


class ParseResult:
    def __init__(self, parsed_expression: ExpressionBase, error: str):
        self.parsed_expression: ExpressionBase = parsed_expression
        self.error: str = error


class Parser:
    def __init__(self):
        self.tokenizer = Tokenizer()
        self.read_next_token()

    # Expression => Term ( ( '+' | '-' ) Term )*
    def parse_expression(self) -> ParseResult:
        initial = self.parse_term()
        error, parsed_expression = initial.error, initial.parsed_expression
        del initial

        if parsed_expression:
            while self.check(TokenType.PLUS) or self.check(TokenType.MINUS):
                operator = self.consume()  # Get operator
                right_term = self.parse_term()

                if right_term.error:
                    error = right_term.error
                    parsed_expression = None
                    break
                elif right_term.parsed_expression:
                    parsed_expression = Expression(
                        lhs=parsed_expression,
                        op=operator,
                        rhs=right_term.parsed_expression
                    )
                else:
                    error = 'Expected expression after ' + operator.value
                    parsed_expression = None
                    break

        return ParseResult(parsed_expression, error)

    # Term => Exponential ( ( '*' | '/' ) Exponential )*
    def parse_term(self) -> ParseResult:
        initial = self.parse_exponential()
        error, parsed_expression = initial.error, initial.parsed_expression
        del initial

        if parsed_expression:
            while self.check(TokenType.STAR) or self.check(TokenType.SLASH):
                operator = self.consume()  # Get operator
                right_exponential = self.parse_exponential()

                if right_exponential.error:
                    error = right_exponential.error
                    parsed_expression = None
                    break
                elif right_exponential.parsed_expression:
                    parsed_expression = Term(
                        lhs=parsed_expression,
                        op=operator,
                        rhs=right_exponential.parsed_expression
                    )
                else:
                    error = 'Expected expression after ' + operator.value
                    parsed_expression = None
                    break

        return ParseResult(parsed_expression, error)

    # Exponential => Atomic ( '**' Atomic )?
    def parse_exponential(self) -> ParseResult:
        initial = self.parse_atomic()
        error, parsed_expression = initial.error, initial.parsed_expression
        del initial

        if parsed_expression and self.check(TokenType.EXPONENT):
            self.read_next_token()  # Skip **
            exponent = self.parse_atomic()
            if exponent.error:
                error = exponent.error
                parsed_expression = None
            elif exponent.parsed_expression:
                parsed_expression = Exponential(
                    base=parsed_expression,
                    exponent=exponent.parsed_expression
                )
            else:
                error = 'Expected expression after **'
                parsed_expression = None

        return ParseResult(parsed_expression, error)

    # Atomic => NUMBER | ( '-' )? ( Atomic | '(' Expression ')' )
    def parse_atomic(self) -> ParseResult:
        if self.check(TokenType.NUMBER):
            parsed_expression = Atomic(
                is_signed=False,
                is_number=True,
                value=self.consume()
            )
            error = None
        else:
            parsed_expression = None
            error = None
            is_signed = self.check(TokenType.MINUS)
            if is_signed:
                self.read_next_token()  # Skip -

            if self.check(TokenType.LEFT_PARENTHESIS):
                self.read_next_token()  # Skip (
                value = self.parse_expression()
                if value.error:
                    error = value.error
                elif value.parsed_expression:
                    if self.check(TokenType.RIGHT_PARENTHESIS):
                        self.read_next_token()  # Skip )
                        parsed_expression = Atomic(
                            is_signed,
                            is_number=False,
                            value=value.parsed_expression
                        )
                    else:
                        error = 'Expected ) after expression'
                else:
                    error = 'Expected expression after ('
            else:
                atom = self.parse_atomic()
                if atom.error:
                    error = atom.error
                elif atom.parsed_expression:
                    parsed_expression = Atomic(
                        is_signed,
                        is_number=False,
                        value=atom.parsed_expression,
                    )
                elif is_signed:
                    error = 'Expected expression after -'

        return ParseResult(parsed_expression, error)

    def consume(self) -> Token:
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


parse_result: ParseResult = Parser().parse_expression()
error, parsed_expression = parse_result.error, parse_result.parsed_expression
del parse_result
if error:
    print(error, file=stderr)
else:
    print(parsed_expression)
    print(parsed_expression.evaluate())
