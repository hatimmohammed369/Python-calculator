#!/usr/bin/python
# How to run:
# python main.py path_to_input_file

from enum import Enum
from sys import stderr
from re import compile
from typing import override
from abc import ABC, abstractmethod
import math
import argparse


class TokenType(Enum):
    NUMBER = 1
    PLUS = 2
    MINUS = 3
    STAR = 4
    SLASH = 5
    EXPONENT = 6
    LEFT_PARENTHESIS = 7
    RIGHT_PARENTHESIS = 8
    END_OF_LINE = 9
    EQUAL = 10
    NAME = 11
    COMMA = 12


class Token:
    # Constructor
    def __init__(self, ttype: TokenType, value: str, line: int, col: int):
        self.ttype: TokenType = ttype
        self.value: str = value
        self.line: int = line
        self.col: int = col

    def __repr__(self):
        value = f'Token(ttype={self.ttype}, '
        value += f"value='{self.value}', "
        value += f"line='{self.line}', "
        value += f'col={self.col})'
        return value


# Scan input file lines to extract tokens
NUMBERS_PATTERN = compile(r'\d+([.]\d+((e|E)(-|\+)?\d+)?)?')
NAMES_PATTERN = compile(r'[a-zA-Z_]+\w*')


class Tokenizer:
    def __init__(self, input_lines: list[str]):
        self.input_lines: list[str] = input_lines
        self.line: int = 0
        self.col: int = 0
        self.skip_end_of_line: bool = False

    def is_at_end(self):
        return self.line >= len(self.input_lines)

    def get_current_line(self):
        if not self.is_at_end():
            return self.input_lines[self.line]
        else:
            # No more lines, return last line
            return self.input_lines[-1]

    def has_more_lines(self):
        return self.line + 1 < len(self.input_lines)

    def __iter__(self):
        return self

    def __next__(self):
        current_line = self.get_current_line()
        if self.col >= len(current_line):
            while True:
                self.line += 1
                self.col = 0
                self.skip_end_of_line = False
                if self.is_at_end():
                    # no more tokens to generate, stop
                    raise StopIteration
                else:
                    current_line = self.get_current_line()
                    if not current_line.isspace():  # do not use an empty line
                        break

        while self.col < len(current_line):
            match current_line[self.col]:
                case '\n':
                    if not self.skip_end_of_line:
                        read_token = Token(
                            ttype=TokenType.END_OF_LINE,
                            value='\\n',
                            line=self.line,
                            col=self.col
                        )
                        self.col += 1
                        return read_token
                    else:
                        # Jump to next non-empty line
                        while True:
                            self.line += 1
                            self.col = 0
                            self.skip_end_of_line = False
                            if self.is_at_end():
                                # no more tokens to generate, stop
                                raise StopIteration
                            else:
                                current_line = self.get_current_line()
                                if not current_line.isspace():
                                    # we found a non-blank line, stop
                                    break
                        continue

                case '\\':
                    all_whitespaces = True
                    rest = current_line[self.col+1:-1]
                    for c in rest:
                        if not c.isspace():
                            all_whitespaces = False
                            break
                        else:
                            self.col += 1
                    if all_whitespaces:
                        # all characters between \ and the next \n
                        # are all whitespaces
                        self.skip_end_of_line = True
                        if len(rest) == 0:
                            # in case \ is immediately followed by \n
                            self.col += 1
                    else:
                        # A stray \, stop generating tokens
                        raise StopIteration

                case c if not c.isspace():
                    read_token = None
                    if number := NUMBERS_PATTERN.match(current_line, self.col):
                        read_token = Token(
                            ttype=TokenType.NUMBER,
                            value=number.group(),
                            line=self.line,
                            col=self.col
                        )
                    elif name := NAMES_PATTERN.match(current_line, self.col):
                        read_token = Token(
                            ttype=TokenType.NAME,
                            value=name.group(),
                            line=self.line,
                            col=self.col
                        )
                    else:
                        match current_line[self.col]:
                            case '+':
                                read_token = Token(
                                    ttype=TokenType.PLUS,
                                    value='+',
                                    line=self.line,
                                    col=self.col
                                )
                            case '-':
                                read_token = Token(
                                    ttype=TokenType.MINUS,
                                    value='-',
                                    line=self.line,
                                    col=self.col
                                )
                            case '*':
                                if current_line[self.col:self.col+2] == '**':
                                    read_token = Token(
                                        ttype=TokenType.EXPONENT,
                                        value='**',
                                        line=self.line,
                                        col=self.col
                                    )
                                else:
                                    read_token = Token(
                                        ttype=TokenType.STAR,
                                        value='*',
                                        line=self.line,
                                        col=self.col
                                    )
                            case '/':
                                read_token = Token(
                                    ttype=TokenType.SLASH,
                                    value='/',
                                    line=self.line,
                                    col=self.col
                                )
                            case '(':
                                read_token = Token(
                                    ttype=TokenType.LEFT_PARENTHESIS,
                                    value='(',
                                    line=self.line,
                                    col=self.col
                                )
                            case ')':
                                read_token = Token(
                                    ttype=TokenType.RIGHT_PARENTHESIS,
                                    value=')',
                                    line=self.line,
                                    col=self.col
                                )
                            case '=':
                                read_token = Token(
                                    ttype=TokenType.EQUAL,
                                    value='=',
                                    line=self.line,
                                    col=self.col
                                )
                            case ',':
                                read_token = Token(
                                    ttype=TokenType.COMMA,
                                    value=',',
                                    line=self.line,
                                    col=self.col
                                )
                    if read_token:
                        self.col += len(read_token.value)
                        return read_token
                    else:
                        # no token was generated, stop iteration
                        raise StopIteration

                case _:
                    # skip any whitespace character
                    self.col += 1


# Context-Free Grammar
# Statement => ( NAME '=' )? Expression END_OF_LINE
# Expression => Term ( ( '+' | '-' ) Term )*
# Term => Exponential ( ( '*' | '/' ) Exponential )*
# Exponential => Atomic ( '**' Atomic )?
# Atomic => NUMBER | NAME | '(' Expression ')' | FunctionCall | '-' Atomic
# FunctionCall => NAME '(' ( Expression ',' )* ')'


MATHEMATICAL_CONSTANTS = {'PI', 'E', 'TAU', 'INFINITY', 'NaN'}
names: dict[str, float] = {}
names['PI'] = math.pi
names['E'] = math.e
names['TAU'] = math.tau
names['INFINITY'] = math.inf
names['NaN'] = math.nan


class ExpressionBase(ABC):
    @abstractmethod
    def evaluate(self):
        pass

    @abstractmethod
    def __repr__(self) -> str:
        pass


class RedefiningConstantError(Exception):
    def __init__(self, constant_token: Token):
        self.constant_token: Token = constant_token


class Statement(ExpressionBase):
    def __init__(self, name_token: Token, expression: ExpressionBase):
        self.name_token: Token = name_token
        self.expression: ExpressionBase = expression

    # Statement => ( NAME '=' )? Expression END_OF_LINE
    @override
    def evaluate(self):
        value = self.expression.evaluate()
        if self.name_token:
            if self.name_token.value not in MATHEMATICAL_CONSTANTS:
                names[self.name_token.value] = value
            else:
                raise RedefiningConstantError(constant_token=self.name_token)
        return value

    @override
    def __repr__(self):
        value = f'{self.expression}'
        if self.name_token:
            value = f'{self.name_token.value} = {value}'
        return value


class Expression(ExpressionBase):
    def __init__(self, lhs, op, rhs):
        self.left_term: ExpressionBase = lhs
        self.operator: Token = op
        self.right_term: ExpressionBase = rhs

    # Expression => Term ( ( '+' | '-' ) Term )*
    @override
    def evaluate(self):
        left = self.left_term.evaluate()
        right = self.right_term.evaluate()
        if self.operator.value == '+':
            return left + right
        else:
            return left - right

    @override
    def __repr__(self) -> str:
        return f'{self.left_term} {self.operator.value} {self.right_term}'


class DivisionByZeroError(Exception):
    def __init__(self, operator: Token):
        self.operator = operator


class Term(ExpressionBase):
    def __init__(self, lhs, op, rhs):
        self.left_exponential: ExpressionBase = lhs
        self.operator: Token = op
        self.right_exponential: ExpressionBase = rhs

    # Term => Exponential ( ( '*' | '/' ) Exponential )*
    @override
    def evaluate(self):
        left = self.left_exponential.evaluate()
        right = self.right_exponential.evaluate()
        if self.operator.value == '*':
            return left * right
        else:
            if right:
                return left / right
            else:
                raise DivisionByZeroError(operator=self.operator)

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
    def evaluate(self):
        base = self.base.evaluate()
        exponent = self.exponent.evaluate()
        return base ** exponent

    @override
    def __repr__(self) -> str:
        return f'{self.base} ** {self.exponent}'


class AtomicType(Enum):
    NUMBER = 1
    NAME = 2
    GROUPED_EXPRESSION = 3
    FUNCTION_CALL = 4


class NameLookupError(Exception):
    def __init__(self, name_token: Token):
        self.name_token = name_token


class Atomic(ExpressionBase):
    def __init__(self, is_signed: bool, atomic_type: AtomicType, value):
        self.is_signed: bool = is_signed
        self.atomic_type: AtomicType = atomic_type
        self.value: Token | ExpressionBase = value

    # Atomic => NUMBER | NAME | '(' Expression ')' | '-' Atomic
    @override
    def evaluate(self):
        match self.atomic_type:
            case AtomicType.NUMBER:
                if '.' in self.value.value:
                    value = float(self.value.value)
                else:
                    value = int(self.value.value)
            case AtomicType.NAME:
                try:
                    value = names[self.value.value]
                except KeyError:
                    raise NameLookupError(name_token=self.value)
            case AtomicType.GROUPED_EXPRESSION:
                value = self.value.evaluate()
        if self.is_signed:
            value *= -1
        return value

    @override
    def __repr__(self) -> str:
        match self.atomic_type:
            case AtomicType.NUMBER | AtomicType.NAME:
                value = f'{self.value.value}'
            case AtomicType.GROUPED_EXPRESSION:
                value = f'({self.value})'
        if self.is_signed:
            value = f'-{value}'
        return value


class InvalidFunctionCallError(Exception):
    def __init__(
            self,
            function_name_token: Token,
            exception_error_message: str
    ):
        self.function_name_token: Token = function_name_token
        self.error = exception_error_message


class FunctionCall(Atomic):
    def __init__(self, function_name: Token, arguments: list[ExpressionBase]):
        self.is_signed = False
        self.atomic_type = AtomicType.FUNCTION_CALL
        self.value = function_name
        self.arguments = arguments

    # FunctionCall => NAME '(' ( Expression ',' )* ')'
    @override
    def evaluate(self):
        function = eval(f'math.{self.value.value}')
        arguments = [argument.evaluate() for argument in self.arguments]
        try:
            return function(*arguments)
        except TypeError as e:
            raise InvalidFunctionCallError(
                function_name_token=self.value,
                exception_error_message=str(e).replace('math.', 'Function ')
            )

    @override
    def __repr__(self) -> str:
        value = self.value.value + '('
        for argument in self.arguments[:-1]:
            value += f'{argument}, '
        value += f'{self.arguments[-1]})'
        return value


class ParseResult:
    def __init__(self, parsed_expression: ExpressionBase, error: str):
        self.parsed_expression: ExpressionBase = parsed_expression
        self.error: str = error


class Parser:
    def __init__(self, input_lines: list[str]):
        self.tokenizer = Tokenizer(input_lines)
        self.read_next_token()

    def read_next_token(self):
        try:
            self.current: Token = next(self.tokenizer)
        except StopIteration:
            self.current: Token = None

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

    def get_token_line(self, token: Token) -> str:
        if token:
            return self.tokenizer.input_lines[token.line]
        else:
            return self.tokenizer.input_lines[-1]

    def __iter__(self):
        return self

    def __next__(self):
        if self.current:
            result = self.parse_statement()
            error, parsed_expression = result.error, result.parsed_expression
            del result
            if parsed_expression:
                if error:
                    raise Exception(
                        "Error & parsed expression\n" +
                        f"error: {error}\n" +
                        f"parsed expression: {parsed_expression}\n"
                    )
                else:
                    # Parsing successful, no errors
                    if self.current and not self.check(TokenType.END_OF_LINE):
                        # Invalid syntax, unexpected item
                        parsed_expression = None
                        error = f'Error in line {self.current.line+1}: '
                        error += 'Invalid syntax, unexpected item\n'
                        error += self.tokenizer.input_lines[self.current.line]
                        error += (' ' * self.current.col)
                        error += ('^' * len(self.current.value))
                    elif self.check(TokenType.END_OF_LINE):
                        self.read_next_token()  # Skip end of line
                    return ParseResult(parsed_expression, error)
            else:
                if error:
                    # A parsing (syntax) error occurred
                    # parsed_expression is None
                    return ParseResult(parsed_expression, error)
                else:
                    # no statement was parsed, stop iteration
                    raise StopIteration
        else:
            # no more tokens to parse, stop iteration
            raise StopIteration

    # Statement => ( NAME '=' )? Expression END_OF_LINE
    def parse_statement(self) -> ParseResult:
        is_assignment = False
        if self.check(TokenType.NAME):
            name = self.consume()
            if not self.check(TokenType.EQUAL):
                # Rewind to (name) token
                # this is NOT an assignment
                self.current = name
                self.tokenizer.col = name.col + len(name.value)
            else:
                is_assignment = True
                self.read_next_token()  # Skip =
        expression = self.parse_expression()
        if not is_assignment:
            return expression
        else:
            error = expression.error
            parsed_expression = expression.parsed_expression
            del expression
            if error:
                parsed_expression = None
            elif parsed_expression:
                parsed_expression = Statement(
                    name_token=name,
                    expression=parsed_expression
                )
            return ParseResult(parsed_expression, error)

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
                    # Expected expression after + or -
                    parsed_expression = None
                    error = f'Error in line {operator.line+1}: '
                    error += f'Expected expression after {operator.value}\n'
                    current_line = self.get_token_line(operator)
                    error += current_line[:operator.col+1]
                    error += ' ' + current_line[operator.col+1:]
                    error += (' ' * (operator.col+1)) + '^'
                    break
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
                    # Expected expression after + or -
                    parsed_expression = None
                    error = f'Error in line {operator.line+1}: '
                    error += f'Expected expression after {operator.value}\n'
                    current_line = self.get_token_line(operator)
                    error += current_line[:operator.col+1]
                    error += ' ' + current_line[operator.col+1:]
                    error += (' ' * (operator.col+1)) + '^'
                    break
        return ParseResult(parsed_expression, error)

    # Exponential => Atomic ( '**' Atomic )?
    def parse_exponential(self) -> ParseResult:
        initial = self.parse_atomic()
        error, parsed_expression = initial.error, initial.parsed_expression
        del initial
        if parsed_expression and self.check(TokenType.EXPONENT):
            operator = self.consume()
            exponent = self.parse_atomic()
            if exponent.error:
                parsed_expression = None
                error = exponent.error
            elif exponent.parsed_expression:
                parsed_expression = Exponential(
                    base=parsed_expression,
                    exponent=exponent.parsed_expression
                )
            else:
                # Expected expression after **
                parsed_expression = None
                error = f'Error in line {operator.line+1}: '
                error += 'Expected expression after **'
                current_line = self.get_token_line(operator)
                error += current_line[:operator.col+2]
                error += ' ' + current_line[operator.col+1:]
                error += (' ' * (operator.col+1)) + '^^'
        return ParseResult(parsed_expression, error)

    # Atomic => NUMBER | NAME | '(' Expression ')' | '-' Atomic
    def parse_atomic(self) -> ParseResult:
        parsed_expression = None
        error = None
        if self.check(TokenType.NUMBER):
            parsed_expression = Atomic(
                is_signed=False,
                atomic_type=AtomicType.NUMBER,
                value=self.consume()
            )
        elif self.check(TokenType.NAME):
            name = self.consume()
            if not self.check(TokenType.LEFT_PARENTHESIS):
                parsed_expression = Atomic(
                    is_signed=False,
                    atomic_type=AtomicType.NAME,
                    value=name
                )
            else:
                self.read_next_token()  # Skip (
                arguments = []
                while True:
                    expression = self.parse_expression()
                    if expression.error:
                        error = expression.error
                        break
                    elif expression.parsed_expression:
                        arguments.append(expression.parsed_expression)
                        if self.check(TokenType.COMMA):
                            self.read_next_token()  # Skip ,
                            continue
                        elif self.check(TokenType.RIGHT_PARENTHESIS):
                            break
                        else:
                            # Expected ) or , after function argument
                            line = len(self.tokenizer.input_lines) - 1
                            if self.current:
                                line = self.current.line
                            error = f'Error in line {line+1}: '
                            error += 'Expected ) or , after '
                            error += 'function argument\n'
                            current_line = self.tokenizer.input_lines[line]
                            end = len(current_line) - 1
                            for k in range(end, -1, -1):
                                if not current_line[k].isspace():
                                    end = k
                                    break
                            error += current_line[:end+1] + '\n'
                            error += (' ' * (end + 1)) + '^'
                            break
                    else:
                        break
                if not error:
                    if self.check(TokenType.RIGHT_PARENTHESIS):
                        self.read_next_token()  # Skip )
                        parsed_expression = FunctionCall(
                            function_name=name,
                            arguments=arguments
                        )
                    else:
                        # Expected ) after function arguments list
                        line = len(self.tokenizer.input_lines) - 1
                        if self.current:
                            line = self.current.line
                        error = f'Error in line {line+1}: '
                        error += 'Expected ) after function arguments list\n'
                        current_line = self.tokenizer.input_lines[line]
                        col = len(current_line) - 1
                        if self.current:
                            col = self.current.col
                        error += current_line[:col]
                        error += ' ' + current_line[col:]
                        error += (' ' * col) + '^'
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
                        atomic_type=AtomicType.GROUPED_EXPRESSION,
                        value=grouped_expression.parsed_expression
                    )
                else:
                    # Expected ) after expression
                    line = len(self.tokenizer.input_lines) - 1
                    if self.current:
                        line = self.current.line
                    error = f'Error in line {line+1}: '
                    error += 'Expected ) after expression\n'
                    current_line = self.tokenizer.input_lines[line]
                    if self.current:
                        col = self.current.col
                    else:
                        col = len(current_line) - 1
                        for k in range(col, -1, -1):
                            if not current_line[k].isspace():
                                col = k
                                break
                    error += current_line[:col] + ' '
                    error += current_line[col:]
                    error += (' ' * col) + '^'
            else:
                # Expected expression after (
                line = len(self.tokenizer.input_lines) - 1
                if self.current:
                    line = self.current.line
                error = f'Error in line {line+1}: '
                error += 'Expected expression after (\n'
                current_line = self.tokenizer.input_lines[line]
                if self.current:
                    col = self.current.col
                else:
                    col = len(current_line) - 1
                error += current_line[0:col]
                error += ' ' + current_line[col:]
                error += (' ' * col) + '^'
        elif self.check(TokenType.MINUS):
            self.read_next_token()  # Skip -
            atomic = self.parse_atomic()
            if atomic.error:
                error = atomic.error
            elif atomic.parsed_expression:
                parsed_expression = Atomic(
                    is_signed=True,
                    atomic_type=atomic.parsed_expression.atomic_type,
                    value=atomic.parsed_expression
                )
            else:
                # Expected expression
                line = len(self.tokenizer.input_lines) - 1
                if self.current:
                    line = self.current.line
                error = f'Error in line {line+1}: '
                error += 'Expected expression\n'
                current_line = self.get_token_line(self.current)
                if self.current:
                    col = self.current.col
                else:
                    col = len(current_line) - 1
                error += current_line[0:col]
                error += ' ' + current_line[col:]
                error += (' ' * col) + '^'
        elif self.check(TokenType.END_OF_LINE):
            # Unexpected end of line
            line = len(self.tokenizer.input_lines) - 1
            if self.current:
                line = self.current.line
            error = f'Error in line {line+1}: '
            error += f'Unexpected end of line {self.tokenizer.line+1}\n'
            current_line = self.tokenizer.input_lines[line]
            end = len(current_line) - 1
            for k in range(end, -1, -1):
                if not current_line[k].isspace():
                    end = k
                    break
            error += current_line[:end+1] + '\n'
            error += (' ' * (end + 1)) + '^'
        return ParseResult(parsed_expression, error)


def process(parser: Parser, interactive: bool):
    for parse_result in parser:
        error = parse_result.error
        parsed_expression = parse_result.parsed_expression
        del parse_result
        if error:
            print(error, file=stderr)
            break
        elif parsed_expression:
            try:
                print(
                    parsed_expression,
                    parsed_expression.evaluate(),
                    sep='\n'
                )
                print()
                continue
            except DivisionByZeroError as e:
                # Division by zero
                error = f'Error in line {e.operator.line+1}: '
                error += 'Division by zero\n'
                error += parser.tokenizer.input_lines[e.operator.line]
                error += (' ' * e.operator.col) + '^'
            except NameLookupError as e:
                # Variable not found
                error = f'Error in line {e.name_token.line+1}: '
                error += f"Variable '{e.name_token.value}' not defined\n"
                error += parser.tokenizer.input_lines[e.name_token.line]
                error += (' ' * e.name_token.col)
                error += ('^' * len(e.name_token.value))
            except RedefiningConstantError as e:
                # Attempting to redefine mathematical constant
                error = f'Error in line {e.constant_token.line+1}: '
                error += 'Attempting to redefine mathematical constant '
                error += f"'{e.constant_token.value}'\n"
                error += parser.tokenizer.input_lines[e.constant_token.line]
                error += (' ' * e.constant_token.col)
                error += ('^' * len(e.constant_token.value))
            except InvalidFunctionCallError as e:
                # Invalid function call
                error = f'Error in line {e.function_name_token.line+1}: '
                error += f'{e.error}\n'
                error += parser.tokenizer.input_lines[
                    e.function_name_token.line
                ]
                error += ' ' * e.function_name_token.col
                error += '^' * len(e.function_name_token.value)
            print(error, file=stderr)
            if not interactive:
                exit(1)


args_parser = argparse.ArgumentParser()
args_parser.add_argument('-f', '--file', help='input file to process')
args = args_parser.parse_args()
if args.file:
    process(
        parser=Parser(
            input_lines=open(args.file, 'r').readlines()
        ),
        interactive=False
    )
else:
    while True:
        try:
            line = input("> ") + '\n'
            process(
                parser=Parser(input_lines=[line]),
                interactive=True
            )
        except EOFError:
            exit(1)
