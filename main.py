#!/usr/bin/python

from enum import Enum
from sys import stderr
from re import compile
from typing import override
from abc import ABC, abstractmethod
from dataclasses import dataclass
import math
import argparse
import operator
import readline


class TokenType(Enum):
    INTEGER = 0
    FLOAT = 1
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
    DOUBLE_SLASH = 13
    PERCENT = 14
    KEYWORD_FN = 15


class Token:
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


NUMBERS_PATTERN = compile(r'\d+(\.\d+)?([Ee][+-]?\d+)?')
NAMES_PATTERN = compile(r'[a-zA-Z_]\w*')


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
                            ttype=(
                                TokenType.FLOAT if number.group(1)
                                else TokenType.INTEGER
                            ),
                            value=number.group(),
                            line=self.line,
                            col=self.col
                        )
                    elif (
                        current_line[self.col:self.col+2] == 'fn' and
                        self.col+2 < len(current_line) and
                        not current_line[self.col+2].isalnum() and
                        not current_line[self.col+2] == '_'
                    ):
                        read_token = Token(
                            ttype=TokenType.KEYWORD_FN,
                            value='fn',
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
                                if current_line[self.col:self.col+2] == '//':
                                    read_token = Token(
                                        ttype=TokenType.DOUBLE_SLASH,
                                        value='//',
                                        line=self.line,
                                        col=self.col
                                    )
                                else:
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
                            case '%':
                                read_token = Token(
                                    ttype=TokenType.PERCENT,
                                    value='%',
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
# Program => ( ( Statement | FunctionDefinition ) END_OF_LINE )*
# Statement  => ( NAME '=' )? Expression
# FunctionDefinition => 'fn' NAME '(' ( NAME ',' )* ')' '=' Expression
# Expression => Term ( ( '+' | '-' ) Term )*
# Term => Exponential ( ( '*' | '/' | '//' | '%' ) Exponential )*
# Exponential => Unary ( '**' Unary )*
# Unary => ( '-' )? Primary
# Primary => FunctionCall | Grouped | Name | Number
# Name => NAME
# Number => INTEGER | FLOAT
# Grouped => '(' Expression ')'
# FunctionCall => NAME '(' ( Expression ',' )* ')'


defined_functions = {}


MATHEMATICAL_CONSTANTS = {'pi', 'e', 'tau', 'infinity', 'nan'}
type Namespace = dict[str, float]
global_namespace: Namespace = {}
global_namespace['pi'] = math.pi
global_namespace['e'] = math.e
global_namespace['tau'] = math.tau
global_namespace['infinity'] = math.inf
global_namespace['nan'] = math.nan
active_function_namespace: list[Namespace] = []


def create_new_namespace():
    active_function_namespace.append({})


def destroy_active_namespace():
    active_function_namespace.pop()


def get_active_namespace() -> Namespace:
    if active_function_namespace:
        return active_function_namespace[-1]
    return global_namespace


OPERATORS_MAP = {
    TokenType.PLUS: operator.add,
    TokenType.MINUS: operator.sub,
    TokenType.STAR: operator.mul,
    TokenType.SLASH: operator.truediv,
    TokenType.EXPONENT: operator.pow,
    TokenType.DOUBLE_SLASH: operator.floordiv,
    TokenType.PERCENT: operator.mod,
}

OPERATORS_STRINGS = {
    TokenType.PLUS: '+',
    TokenType.MINUS: '-',
    TokenType.STAR: '*',
    TokenType.SLASH: '/',
    TokenType.EXPONENT: '**',
    TokenType.DOUBLE_SLASH: '//',
    TokenType.PERCENT: '%',
}


class ExpressionAST(ABC):
    @abstractmethod
    def evaluate(self):
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass

    @abstractmethod
    def __repr__(self) -> str:
        pass


class RedefiningConstantError(Exception):
    def __init__(self, constant_token: Token):
        self.constant_token: Token = constant_token


# Statement  => ( NAME '=' )? Expression
class Statement(ExpressionAST):
    def __init__(self, name_token: Token, expression: ExpressionAST):
        self.name_token: Token = name_token
        self.expression: ExpressionAST = expression

    # Statement  => ( NAME '=' )? Expression
    @override
    def evaluate(self):
        value = self.expression.evaluate()
        if var := (
            None if not self.name_token
            else self.name_token.value
        ):
            if var not in MATHEMATICAL_CONSTANTS:
                global_namespace[var] = value
            else:
                raise RedefiningConstantError(
                    constant_token=self.name_token
                )
        return value

    @override
    def __str__(self) -> str:
        return f'{self.name_token.value} = {self.expression}'

    @override
    def __repr__(self) -> str:
        return (
            f'Statement(name_token={repr(self.name_token)}, ' +
            f'expression={repr(self.expression)})'
        )


# FunctionDefinition => 'fn' NAME '(' ( NAME ',' )* ')' '=' Expression
class FunctionDefinition(ExpressionAST):
    def __init__(
        self, name: Token, parameters: list[Token], body: ExpressionAST
    ):
        self.name: Token = name
        self.parameters: list[Token] = parameters
        self.body: ExpressionAST = body

    @override
    def evaluate(self):
        defined_functions[self.name.value] = self
        parameters = ', '.join(p.value for p in self.parameters)
        return f'function {self.name.value}({parameters})'

    @override
    def __str__(self) -> str:
        parameters = ', '.join(
            parameter.value
            for parameter in self.parameters
        )
        return f'{self.name.value}({parameters}) = {self.body}'

    @override
    def __repr__(self) -> str:
        return (
            f'FunctionDefinition(name={repr(self.name)}, ' +
            f'parameters={[repr(p) for p in self.parameters]}, ' +
            f'body={repr(self.body)})'
        )


# Expression => Term ( ( '+' | '-' ) Term )*
class Expression(ExpressionAST):
    def __init__(
        self,
        terms: list[ExpressionAST],
        operators: list[TokenType]
    ):
        self.terms: list[ExpressionAST] = terms
        self.operators: list[TokenType] = operators

    # Expression => Term ( ( '+' | '-' ) Term )*
    @override
    def evaluate(self):
        value = self.terms[0].evaluate()
        for i in range(1, len(self.terms)):
            term = self.terms[i].evaluate()
            op = OPERATORS_MAP[self.operators[i-1]]
            value = op(value, term)
        return value

    @override
    def __str__(self) -> str:
        terms = [str(term) for term in self.terms]
        operators = [
            OPERATORS_STRINGS[op]
            for op in self.operators
        ] + ['']
        return ''.join(
            f'{term} {op} '
            for (term, op) in zip(terms, operators)
        )

    @override
    def __repr__(self) -> str:
        terms = [repr(term) for term in self.terms]
        operators = [repr(op) for op in self.operators]
        return f'Expression(terms={repr(terms)}, operators={repr(operators)})'


class DivisionByZeroError(Exception):
    def __init__(self, operator: Token):
        self.operator = operator


class ZeroModulusError(Exception):
    def __init__(self, operator: Token):
        self.operator = operator


# Term => Exponential ( ( '*' | '/' | '//' | '%' ) Exponential )*
class Term(ExpressionAST):
    def __init__(
        self,
        exponentials: list[ExpressionAST],
        operators: list[Token]
    ):
        self.exponentials: list[ExpressionAST] = exponentials
        self.operators: list[Token] = operators

    # Term => Exponential ( ( '*' | '/' | '//' | '%' ) Exponential )*
    @override
    def evaluate(self):
        value = self.exponentials[0].evaluate()
        for i in range(1, len(self.exponentials)):
            exponential = self.exponentials[i].evaluate()
            op_tok = self.operators[i-1]
            op = OPERATORS_MAP[op_tok.ttype]
            if not exponential:
                match op_tok.ttype:
                    case TokenType.SLASH | TokenType.DOUBLE_SLASH:
                        raise DivisionByZeroError(operator=op_tok)
                    case TokenType.PERCENT:
                        raise ZeroModulusError(operator=op_tok)
            value = op(value, exponential)
        return value

    @override
    def __str__(self) -> str:
        exponentials = [
            str(exponential)
            for exponential in self.exponentials
        ]
        operators = [
            OPERATORS_STRINGS[op.ttype]
            for op in self.operators
        ] + ['']
        return ''.join(
            f'{exponential} {op} '
            for (exponential, op) in zip(exponentials, operators)
        )

    @override
    def __repr__(self) -> str:
        exponentials = [
            repr(exponential)
            for exponential in self.exponentials
        ]
        operators = [repr(op.ttype) for op in self.operators]
        return (
            'Term(exponentials=' +
            f'{repr(exponentials)}, ' +
            f'operators={repr(operators)})'
        )


# Exponential => Unary ( '**' Unary )*
class Exponential(ExpressionAST):
    def __init__(self, unaries: list[ExpressionAST]):
        self.unaries: list[ExpressionAST] = unaries

    # Exponential => Unary ( '**' Unary )*
    @override
    def evaluate(self):
        value = self.unaries[-1].evaluate()
        for left in reversed(self.unaries[:-1]):
            left = left.evaluate()
            value = left ** value
        return value

    @override
    def __str__(self) -> str:
        return '** '.join(str(unary) for unary in self.unaries)

    @override
    def __repr__(self) -> str:
        unaries = [
            repr(unary)
            for unary in self.unaries
        ]
        return f'Exponential(unaries={unaries})'


# Unary => ( '-' )? Primary
class Unary(ExpressionAST):
    def __init__(self, sign_token: Token, expression: ExpressionAST):
        self.sign_token: Token = sign_token
        self.expression: ExpressionAST = expression

    # Unary => ( '-' )? Primary
    @override
    def evaluate(self):
        return -self.expression.evaluate()

    @override
    def __str__(self) -> str:
        return f'-{self.expression}'

    @override
    def __repr__(self) -> str:
        return (
            f'Unary(sign_token={repr(self.sign_token)}, ' +
            f'expression={repr(self.expression)})'
        )


# Primary => FunctionCall | Grouped | NAME | Number
class Primary(ExpressionAST):
    @abstractmethod
    def evaluate(self):
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass

    @abstractmethod
    def __repr__(self) -> str:
        pass


class NameLookupError(Exception):
    def __init__(self, name_token: Token):
        self.name_token = name_token


# Name => NAME
class Name(Primary):
    def __init__(self, name_token: Token):
        self.name: Token = name_token

    # Name => NAME
    @override
    def evaluate(self):
        if active_function_namespace:
            if value := get_active_namespace().get(self.name.value, None):
                return value
        try:
            return global_namespace[self.name.value]
        except KeyError:
            raise NameLookupError(name_token=self.name)

    @override
    def __str__(self) -> str:
        return self.name.value

    @override
    def __repr__(self) -> str:
        return f'Name(name={repr(self.name)})'


# Number => INTEGER | FLOAT
class Number(Primary):
    def __init__(self, number_token: Token):
        self.number: Token = number_token

    @override
    def evaluate(self):
        match self.number.ttype:
            case TokenType.INTEGER:
                return int(self.number.value)
            case TokenType.FLOAT:
                return float(self.number.value)

    @override
    def __str__(self) -> str:
        return self.number.value

    @override
    def __repr__(self) -> str:
        return f'Number(number={repr(self.number)})'


# Grouped => '(' Expression ')'
class Grouped(Primary):
    def __init__(self, expression: ExpressionAST):
        self.expression: ExpressionAST = expression

    # Grouped => '(' Expression ')'
    @override
    def evaluate(self):
        return self.expression.evaluate()

    @override
    def __str__(self):
        return f'({self.expression})'

    @override
    def __repr__(self):
        return f'Grouped(expression={repr(self.expression)})'


class InvalidFunctionCallError(Exception):
    def __init__(
            self,
            function_name_token: Token,
            exception_error_message: str
    ):
        self.function_name_token: Token = function_name_token
        self.error = exception_error_message


# FunctionCall => NAME '(' ( Expression ',' )* ')'
class FunctionCall(Primary):
    def __init__(self, function_name: Token, arguments: list[ExpressionAST]):
        self.function_name: Token = function_name
        self.arguments: list[ExpressionAST] = arguments

    # FunctionCall => NAME '(' ( Expression ',' )* ')'
    @override
    def evaluate(self):
        arguments = [argument.evaluate() for argument in self.arguments]
        function: FunctionDefinition = (
            defined_functions.get(self.function_name.value, None)
        )
        if function and len(function.parameters) == len(arguments):
            create_new_namespace()
            get_active_namespace().update({
                paramter.value: argument
                for (paramter, argument) in zip(function.parameters, arguments)
            })
            value = function.body.evaluate()
            destroy_active_namespace()
            return value
        elif function := getattr(math, self.function_name.value, None):
            try:
                return function(*arguments)
            except:
                pass

        raise InvalidFunctionCallError(
            function_name_token=self.function_name,
            exception_error_message=(
                f'No such function {self.function_name.value}'
            )
        )

    @override
    def __str__(self):
        args = ', '.join(str(a) for a in self.arguments)
        return f'{self.function_name.value}({args})'

    @override
    def __repr__(self) -> str:
        return (
            f'FunctionCall(function_name={repr(self.function_name)}, ' +
            f'arguments={[repr(arg) for arg in self.arguments]})'
        )


@dataclass(init=True, repr=True, frozen=True)
class ParseResult:
    parsed_expression: ExpressionAST
    error: str

    def __iter__(self):
        yield self.parsed_expression
        yield self.error


class Parser:
    def __init__(self, input_lines: list[str]):
        self.tokenizer = Tokenizer(input_lines)
        self.read_next_token()

    def read_next_token(self):
        try:
            self.current: Token = next(self.tokenizer)
        except StopIteration:
            self.current: Token = None

    def consume_if(self, *expected_types: TokenType) -> Token:
        if self.check(*expected_types):
            return self.consume()
        return None

    def consume(self) -> Token:
        copy = self.current
        self.read_next_token()
        return copy

    def check(self, *expected_types: TokenType) -> bool:
        if self.current:
            for t in expected_types:
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

    # Program => ( ( Statement | FunctionDefinition ) END_OF_LINE )*
    # Statement  => ( NAME '=' )? Expression
    # FunctionDefinition => 'fn' NAME '(' ( NAME ',' )* ')' '=' Expression
    # Expression => Term ( ( '+' | '-' ) Term )*
    def __next__(self):
        if self.current:
            match self.current.ttype:
                case TokenType.KEYWORD_FN:
                    result = self.parse_function_definition()
                case _:
                    result = self.parse_statement()
            parsed_expression, error = result
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
                        self.read_next_token()  # skip end of line
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

    # Statement  => ( NAME '=' )? Expression
    def parse_statement(self) -> ParseResult:
        parsed_expression, error = self.parse_expression()
        if not error:
            if (
                isinstance(parsed_expression, Name) and
                self.consume_if(TokenType.EQUAL)
            ):
                name = parsed_expression.name
                parsed_expression, error = self.parse_expression()
                if not error:
                    parsed_expression = Statement(
                        name_token=name,
                        expression=parsed_expression
                    )
                else:
                    parsed_expression = None
        else:
            parsed_expression = None
        return ParseResult(parsed_expression, error)

    # FunctionDefinition => 'fn' NAME '(' ( NAME ',' )* ')' '=' Expression
    def parse_function_definition(self) -> ParseResult:
        self.read_next_token()  # skip keyword fn
        if not (function_name := self.consume_if(TokenType.NAME)):
            # Syntax error, expected function name
            parsed_expression = None
            error = f'Error in line {function_name.line+1}: '
            error += 'Expected function name after keyword fn\n'
            current_line = self.get_token_line(function_name)
            error += current_line[:function_name.col+1]
            error += ' ' + current_line[function_name.col+1:]
            error += (' ' * (function_name.col+1)) + '^'
            return ParseResult(parsed_expression, error)
        if not (parenthesis := self.consume_if(TokenType.LEFT_PARENTHESIS)):
            # Syntax error, expected ( after function name
            parsed_expression = None
            error = f'Error in line {parenthesis.line+1}: '
            error += 'Expected ( after function name\n'
            current_line = self.get_token_line(parenthesis)
            error += current_line[:parenthesis.col+1]
            error += ' ' + current_line[parenthesis.col+1:]
            error += (' ' * (parenthesis.col+1)) + '^'
            return ParseResult(parsed_expression, error)
        parameters: list[Token] = []
        while parameter := self.consume_if(TokenType.NAME):
            parameters.append(parameter)
            self.consume_if(TokenType.COMMA)
        if not (parenthesis := self.consume_if(TokenType.RIGHT_PARENTHESIS)):
            # Syntax error, expected ) after parameters list
            parsed_expression = None
            error = f'Error in line {parenthesis.line+1}: '
            error += 'Expected ) after parameters list\n'
            current_line = self.get_token_line(parenthesis)
            error += current_line[:parenthesis.col+1]
            error += ' ' + current_line[parenthesis.col+1:]
            error += (' ' * (parenthesis.col+1)) + '^'
            return ParseResult(parsed_expression, error)
        self.read_next_token()  # skip =
        parsed_expression, error = self.parse_expression()
        if error:
            parsed_expression = None
        else:
            parsed_expression = FunctionDefinition(
                function_name,
                parameters,
                parsed_expression
            )
        return ParseResult(parsed_expression, error)

    # Expression => Term ( ( '+' | '-' ) Term )*
    def parse_expression(self) -> ParseResult:
        parsed_expression, error = self.parse_term()
        if parsed_expression:
            terms: list[ExpressionAST] = [parsed_expression]
            operators: list[TokenType] = []
            while not error and self.check(TokenType.PLUS, TokenType.MINUS):
                operator = self.consume()  # Get operator
                operators.append(operator.ttype)
                parsed_expression, error = self.parse_term()
                if error:
                    parsed_expression = None
                elif parsed_expression:
                    terms.append(parsed_expression)
                else:
                    # Expected expression after + or -
                    parsed_expression = None
                    error = f'Error in line {operator.line+1}: '
                    error += f'Expected expression after {operator.value}\n'
                    current_line = self.get_token_line(operator)
                    error += current_line[:operator.col+1]
                    error += ' ' + current_line[operator.col+1:]
                    error += (' ' * (operator.col+1)) + '^'
            if not error:
                if len(terms) == 1:
                    parsed_expression = terms.pop()
                else:
                    parsed_expression = Expression(terms, operators)
        return ParseResult(parsed_expression, error)

    # Term => Exponential ( ( '*' | '/' | '//' | '%' ) Exponential )*
    def parse_term(self) -> ParseResult:
        parsed_expression, error = self.parse_exponential()
        if parsed_expression:
            exponentials: list[ExpressionAST] = [parsed_expression]
            operators: list[Token] = []
            while not error and self.check(
                TokenType.STAR, TokenType.SLASH,
                TokenType.DOUBLE_SLASH, TokenType.PERCENT
            ):
                operator = self.consume()  # Get operator
                operators.append(operator)
                parsed_expression, error = self.parse_exponential()
                if error:
                    parsed_expression = None
                elif parsed_expression:
                    exponentials.append(parsed_expression)
                else:
                    # Expected expression after * / // %
                    parsed_expression = None
                    error = f'Error in line {operator.line+1}: '
                    error += f'Expected expression after {operator.value}\n'
                    current_line = self.get_token_line(operator)
                    error += current_line[:operator.col+1]
                    error += ' ' + current_line[operator.col+1:]
                    error += (' ' * (operator.col+1)) + '^'
            if not error:
                if len(exponentials) == 1:
                    parsed_expression = exponentials.pop()
                else:
                    parsed_expression = Term(exponentials, operators)
        return ParseResult(parsed_expression, error)

    # Exponential => Unary ( '**' Unary )*
    def parse_exponential(self) -> ParseResult:
        parsed_expression, error = self.parse_unary()
        if parsed_expression:
            items: list[ExpressionAST] = [parsed_expression]
            while not error and self.check(TokenType.EXPONENT):
                operator = self.consume()
                parsed_expression, error = self.parse_unary()
                if error:
                    parsed_expression = None
                elif parsed_expression:
                    items.append(parsed_expression)
                else:
                    # Expected expression after **
                    parsed_expression = None
                    error = f'Error in line {operator.line+1}: '
                    error += 'Expected expression after **'
                    current_line = self.get_token_line(operator)
                    error += current_line[:operator.col+2]
                    error += ' ' + current_line[operator.col+1:]
                    error += (' ' * (operator.col+1)) + '^^'
            if not error and len(items) > 1:
                parsed_expression = Exponential(unaries=items)
        return ParseResult(parsed_expression, error)

    # Unary => ( '-' )? Primary
    def parse_unary(self) -> ParseResult:
        sign = self.consume_if(TokenType.MINUS)
        parsed_expression, error = self.parse_primary()
        if not error and sign:
            if not parsed_expression:
                # Expected expression after -
                current_line = self.tokenizer.input_lines[sign.line]
                error += f'Error in line {sign.line}: '
                error += 'Expected expression after -\n'
                error += current_line[:sign.col+1]
                if not current_line[sign.col+1:sign.col+2].isspace():
                    error += ' '
                error += current_line[sign.col:]
                error += ' ' * sign.col + '^'
            else:
                parsed_expression = Unary(
                    sign_token=sign,
                    expression=parsed_expression
                )
        return ParseResult(parsed_expression, error)

    # Primary => FunctionCall | Grouped | NAME | Number
    def parse_primary(self) -> ParseResult:
        parsed_expression = None
        error = None
        if self.check(TokenType.INTEGER, TokenType.FLOAT):
            parsed_expression = Number(number_token=self.consume())
        elif self.check(TokenType.NAME):
            name = self.consume()
            if not self.check(TokenType.LEFT_PARENTHESIS):
                parsed_expression = Name(name_token=name)
            else:
                parsed_expression, error = (
                    self.parse_function_call(function_name=name)
                )
        elif self.check(TokenType.LEFT_PARENTHESIS):
            parsed_expression, error = (
                self.parse_grouped_expression()
            )
        elif self.check(TokenType.END_OF_LINE):
            # Unexpected end of line
            line = self.current.line
            error = f'Error in line {line+1}: '
            error += f'Unexpected end of line {line+1}\n'
            current_line = self.tokenizer.input_lines[line]
            end = len(current_line) - 1
            for k in range(end, -1, -1):
                if not current_line[k].isspace():
                    end = k
                    break
            error += current_line[:end+1] + '\n'
            error += (' ' * (end + 1)) + '^'
        return ParseResult(parsed_expression, error)

    # Grouped => '(' Expression ')'
    def parse_grouped_expression(self) -> ParseResult:
        self.read_next_token()  # skip opening (
        parsed_expression, error = self.parse_expression()
        if error:
            parsed_expression = None
        elif parsed_expression:
            if self.check(TokenType.RIGHT_PARENTHESIS):
                self.read_next_token()  # skip closing )
                if not isinstance(parsed_expression, Primary):
                    # Do not group a primary expression
                    # (number, name, group, function call)
                    parsed_expression = Grouped(
                        expression=parsed_expression
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
        return ParseResult(parsed_expression, error)

    # FunctionCall => NAME '(' ( Expression ',' )* ')'
    def parse_function_call(self, function_name: Token) -> ParseResult:
        self.read_next_token()  # skip opening (
        parsed_expression = None
        error = None
        arguments = []
        while not error:
            parsed_expression, error = self.parse_expression()
            if error:
                parsed_expression = None
            elif parsed_expression:
                arguments.append(parsed_expression)
                if self.check(TokenType.COMMA):
                    self.read_next_token()  # skip ,
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
            else:
                break
        if not error:
            if self.check(TokenType.RIGHT_PARENTHESIS):
                self.read_next_token()  # skip closing )
                parsed_expression = FunctionCall(
                    function_name, arguments
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
        return ParseResult(parsed_expression, error)


def process(parser: Parser, interactive: bool):
    for parse_result in parser:
        parsed_expression, error = parse_result
        if error:
            print(error, file=stderr)
            break
        elif parsed_expression:
            try:
                print(
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
            except ZeroModulusError as e:
                # Integer modulo by zero
                error = f'Error in line {e.operator.line+1}: '
                error += 'Integer modulo by zero\n'
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
            if line := input("> "):
                line += '\n'
                process(
                    parser=Parser(input_lines=[line]),
                    interactive=True
                )
        except (EOFError, KeyboardInterrupt):
            exit(1)
