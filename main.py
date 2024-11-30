# How to run:
# python main.py path_to_input_file

# 1 - Implement basic arithmetic
# + - * / ** -(Expression)
# 2 - Implement nested expressions
# 3 - Implement multi-lined processing
# (\n to end expression, \ to continue expression in next line)
# 4 - Implement variables
# 5 - Implement function calls

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
    def __init__(self, ttype: TokenType, value: str, col: int):
        self.ttype: TokenType = ttype
        self.value: str = value
        self.col: int = col

    def __repr__(self):
        value = f'Token(ttype={self.ttype}, '
        value += f"value='{self.value}', "
        value += f'col={self.col})'
        return value


# Scan input file lines to extract tokens
NUMBERS_PATTERN = compile(r'(-)?\d+([.]\d+((e|E)(-|\+)?\d+)?)?')


class Tokenizer:
    def __init__(self):
        self.input_lines: list[str] = open(argv[1], 'r').readlines()
        self.line: int = 0
        self.pos: int = 0
        self.col: int = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.line < len(self.input_lines):
            current_line = self.input_lines[self.line]

            if self.pos >= len(current_line):
                self.col = 0
                self.pos = 0
                self.line += 1
                current_line = self.input_lines[self.line]

            while self.pos < len(current_line):
                if current_line[self.pos] != ' ':
                    read_token = None
                    if m := NUMBERS_PATTERN.match(current_line, self.pos):
                        read_token = Token(
                            ttype=TokenType.NUMBER,
                            value=m.group(),
                            col=self.col
                        )
                    elif current_line[self.pos] == '+':
                        read_token = Token(
                            ttype=TokenType.PLUS,
                            value='+',
                            col=self.col
                        )
                    elif current_line[self.pos] == '-':
                        read_token = Token(
                            ttype=TokenType.MINUS,
                            value='-',
                            col=self.col
                        )
                    elif current_line[self.pos] == '*':
                        if self.pos+1 < len(current_line) and \
                                current_line[self.pos+1] == '*':
                            read_token = Token(
                                ttype=TokenType.EXPONENT,
                                value='**',
                                col=self.col
                            )
                        else:
                            read_token = Token(
                                ttype=TokenType.STAR,
                                value='*',
                                col=self.col
                            )
                    elif current_line[self.pos] == '/':
                        read_token = Token(
                            ttype=TokenType.SLASH,
                            value='/',
                            col=self.col
                        )
                    elif current_line[self.pos] == '(':
                        read_token = Token(
                            ttype=TokenType.LEFT_PARENTHESIS,
                            value='(',
                            col=self.col
                        )
                    elif current_line[self.pos] == ')':
                        read_token = Token(
                            ttype=TokenType.RIGHT_PARENTHESIS,
                            value=')',
                            col=self.col
                        )

                    if read_token:
                        self.pos += len(read_token.value)
                        self.col += len(read_token.value)
                        return read_token
                    else:
                        break
                else:
                    self.pos += 1
                    self.col += 1
        raise StopIteration


# Context-Free Grammar
# Expression => Term ( ( '+' | '-' ) Term )*
# Term => Exponential ( ( '*' | '/' ) Exponential )*
# Exponential => Atomic ( '**' Atomic )?
# Atomic => NUMBER | '(' Expression ')' | '-' Atomic


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
    def __init__(self, is_signed, is_number, is_grouped, value):
        self.is_signed = is_signed
        self.is_number = is_number
        self.is_grouped = is_grouped
        self.value: Token | ExpressionBase = value

    # Atomic => NUMBER | '(' Expression ')' | '-' Atomic
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
        if self.is_number:
            return self.value.value
        value = f'{self.value}'
        if self.is_grouped:
            value = f'({value})'
        if self.is_signed:
            value = f'-{value}'
        return value


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
            while self.check(TokenType.PLUS, TokenType.MINUS):
                operator = self.consume()  # Get operator
                right_term = self.parse_term()

                if right_term.error:
                    parsed_expression = None
                    error = right_term.error
                    break
                elif right_term.parsed_expression:
                    parsed_expression = Expression(
                        lhs=parsed_expression,
                        op=operator,
                        rhs=right_term.parsed_expression
                    )
                else:
                    parsed_expression = None
                    current_line = self.tokenizer.input_lines[
                        self.tokenizer.line
                    ]
                    col = len(current_line) - 1
                    if self.current:
                        col = self.current.col
                    error = current_line[0:col] + \
                        ' ' + current_line[col:]
                    error += (' ' * col) + '^\n'
                    error += f'Expected expression after {operator.value}'
                    break
            if parsed_expression and self.current:
                parsed_expression = None
                error = self.tokenizer.input_lines[
                    self.tokenizer.line
                ]
                col = self.current.col
                error += (' ' * col) + ('^' * len(self.current.value)) + '\n'
                error += 'Unexpected item'

        return ParseResult(parsed_expression, error)

    # Term => Exponential ( ( '*' | '/' ) Exponential )*
    def parse_term(self) -> ParseResult:
        initial = self.parse_exponential()
        error, parsed_expression = initial.error, initial.parsed_expression
        del initial

        if parsed_expression:
            while self.check(TokenType.STAR, TokenType.SLASH):
                operator = self.consume()  # Get operator
                right_exponential = self.parse_exponential()

                if right_exponential.error:
                    parsed_expression = None
                    error = right_exponential.error
                    break
                elif right_exponential.parsed_expression:
                    parsed_expression = Term(
                        lhs=parsed_expression,
                        op=operator,
                        rhs=right_exponential.parsed_expression
                    )
                else:
                    parsed_expression = None
                    current_line = self.tokenizer.input_lines[
                        self.tokenizer.line
                    ]
                    col = len(current_line) - 1
                    if self.current:
                        col = self.current.col
                    error = current_line[0:col] + \
                        ' ' + current_line[col:]
                    error += (' ' * col) + '^\n'
                    error += f'Expected expression after {operator.value}'
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
                parsed_expression = None
                current_line = self.tokenizer.input_lines[
                    self.tokenizer.line
                ]
                col = len(current_line) - 1
                if self.current:
                    col = self.current.col
                error = current_line[0:col] + \
                    ' ' + current_line[col:]
                error += (' ' * col) + '^\n'
                error += 'Expected expression after **'

        return ParseResult(parsed_expression, error)

    # Atomic => NUMBER | '(' Expression ')' | '-' Atomic
    def parse_atomic(self) -> ParseResult:
        parsed_expression = None
        error = None
        if self.check(TokenType.NUMBER):
            parsed_expression = Atomic(
                is_signed=False,
                is_number=True,
                is_grouped=False,
                value=self.consume()
            )
        elif self.check(TokenType.LEFT_PARENTHESIS):
            self.read_next_token()  # Skip (
            grouped_expression = self.parse_expression()
            if grouped_expression.error:
                error = grouped_expression.error
            elif grouped_expression.parsed_expression:
                if self.check(TokenType.RIGHT_PARENTHESIS):
                    self.read_next_token()  # Skip )
                    parsed_expression = Atomic(
                        is_signed=False,
                        is_number=False,
                        is_grouped=True,
                        value=grouped_expression.parsed_expression
                    )
                else:
                    current_line = self.tokenizer.input_lines[
                        self.tokenizer.line
                    ]
                    col = len(current_line) - 1
                    if self.current:
                        col = self.current.col
                    error = current_line[0:col] + \
                        ' ' + current_line[col:]
                    error += (' ' * col) + '^\n'
                    error += 'Expected ) after expression'
            else:
                current_line = self.tokenizer.input_lines[
                    self.tokenizer.line
                ]
                col = len(current_line) - 1
                if self.current:
                    col = self.current.col
                error = current_line[0:col] + \
                    ' ' + current_line[col:]
                error += (' ' * col) + '^\n'
                error += 'Expected expression ('
        elif self.check(TokenType.MINUS):
            self.read_next_token()  # Skip -
            atomic = self.parse_atomic()
            if atomic.error:
                error = atomic.error
            elif atomic.parsed_expression:
                parsed_expression = Atomic(
                    is_signed=True,
                    is_number=False,
                    is_grouped=False,
                    value=grouped_expression.parsed_expression
                )
            else:
                current_line = self.tokenizer.input_lines[
                    self.tokenizer.line
                ]
                col = len(current_line) - 1
                if self.current:
                    col = self.current.col
                error = current_line[0:col] + \
                    ' ' + current_line[col:]
                error += (' ' * col) + '^\n'
                error += 'Expected expression'

        return ParseResult(parsed_expression, error)

    def consume(self) -> Token:
        copy = self.current
        self.read_next_token()
        return copy

    def check(self, *ttype: TokenType) -> bool:
        if self.current:
            for t in ttype:
                if self.current.ttype == t:
                    return True
        return False

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
