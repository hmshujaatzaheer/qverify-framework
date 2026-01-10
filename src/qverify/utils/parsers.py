"""
Quantum Program Parsers for QVERIFY.

This module provides parsers for various quantum programming languages
including Silq and OpenQASM.
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterator, Optional, Union

logger = logging.getLogger(__name__)


class TokenType(Enum):
    """Token types for quantum program parsing."""
    
    # Keywords
    DEF = "def"
    RETURN = "return"
    IF = "if"
    ELSE = "else"
    FOR = "for"
    WHILE = "while"
    IN = "in"
    
    # Types
    QUBIT = "qubit"
    QUBIT_ARRAY = "qubit[]"
    BOOL = "bool"
    INT = "int"
    
    # Operators
    ASSIGN = "="
    ARROW = "->"
    PLUS = "+"
    MINUS = "-"
    STAR = "*"
    SLASH = "/"
    COLON = ":"
    SEMICOLON = ";"
    COMMA = ","
    DOT = "."
    DOTDOT = ".."
    
    # Brackets
    LPAREN = "("
    RPAREN = ")"
    LBRACE = "{"
    RBRACE = "}"
    LBRACKET = "["
    RBRACKET = "]"
    
    # Literals
    IDENTIFIER = "IDENTIFIER"
    NUMBER = "NUMBER"
    STRING = "STRING"
    
    # Special
    COMMENT = "COMMENT"
    NEWLINE = "NEWLINE"
    EOF = "EOF"


@dataclass
class Token:
    """A token from lexical analysis."""
    
    type: TokenType
    value: str
    line: int
    column: int
    
    def __str__(self) -> str:
        return f"Token({self.type.name}, {self.value!r}, line={self.line})"


class Lexer:
    """Lexical analyzer for quantum programs."""
    
    def __init__(self, source: str) -> None:
        self.source = source
        self.pos = 0
        self.line = 1
        self.column = 1
    
    def tokenize(self) -> Iterator[Token]:
        """Tokenize the source code."""
        while self.pos < len(self.source):
            # Skip whitespace
            if self.source[self.pos].isspace():
                if self.source[self.pos] == '\n':
                    self.line += 1
                    self.column = 1
                else:
                    self.column += 1
                self.pos += 1
                continue
            
            # Skip comments
            if self.source[self.pos:self.pos+2] == '//':
                while self.pos < len(self.source) and self.source[self.pos] != '\n':
                    self.pos += 1
                continue
            
            # Multi-char operators
            if self.source[self.pos:self.pos+2] == '->':
                yield Token(TokenType.ARROW, "->", self.line, self.column)
                self.pos += 2
                self.column += 2
                continue
            
            if self.source[self.pos:self.pos+2] == '..':
                yield Token(TokenType.DOTDOT, "..", self.line, self.column)
                self.pos += 2
                self.column += 2
                continue
            
            # Single-char operators
            char_tokens = {
                '(': TokenType.LPAREN,
                ')': TokenType.RPAREN,
                '{': TokenType.LBRACE,
                '}': TokenType.RBRACE,
                '[': TokenType.LBRACKET,
                ']': TokenType.RBRACKET,
                '=': TokenType.ASSIGN,
                ':': TokenType.COLON,
                ';': TokenType.SEMICOLON,
                ',': TokenType.COMMA,
                '.': TokenType.DOT,
                '+': TokenType.PLUS,
                '-': TokenType.MINUS,
                '*': TokenType.STAR,
                '/': TokenType.SLASH,
            }
            
            if self.source[self.pos] in char_tokens:
                yield Token(
                    char_tokens[self.source[self.pos]],
                    self.source[self.pos],
                    self.line,
                    self.column
                )
                self.pos += 1
                self.column += 1
                continue
            
            # Numbers
            if self.source[self.pos].isdigit():
                start = self.pos
                while self.pos < len(self.source) and self.source[self.pos].isdigit():
                    self.pos += 1
                value = self.source[start:self.pos]
                yield Token(TokenType.NUMBER, value, self.line, self.column)
                self.column += len(value)
                continue
            
            # Identifiers and keywords
            if self.source[self.pos].isalpha() or self.source[self.pos] == '_':
                start = self.pos
                while self.pos < len(self.source) and (
                    self.source[self.pos].isalnum() or self.source[self.pos] == '_'
                ):
                    self.pos += 1
                value = self.source[start:self.pos]
                
                # Check for keywords
                keywords = {
                    'def': TokenType.DEF,
                    'return': TokenType.RETURN,
                    'if': TokenType.IF,
                    'else': TokenType.ELSE,
                    'for': TokenType.FOR,
                    'while': TokenType.WHILE,
                    'in': TokenType.IN,
                    'qubit': TokenType.QUBIT,
                    'bool': TokenType.BOOL,
                    'int': TokenType.INT,
                }
                
                token_type = keywords.get(value, TokenType.IDENTIFIER)
                yield Token(token_type, value, self.line, self.column)
                self.column += len(value)
                continue
            
            # Unknown character
            logger.warning(f"Unknown character: {self.source[self.pos]!r} at line {self.line}")
            self.pos += 1
            self.column += 1
        
        yield Token(TokenType.EOF, "", self.line, self.column)


class Parser(ABC):
    """Abstract base class for quantum program parsers."""
    
    @abstractmethod
    def parse(self, source: str) -> dict[str, Any]:
        """Parse source code into AST."""
        pass


class SilqParser(Parser):
    """
    Parser for Silq quantum programming language.
    
    Example:
        >>> parser = SilqParser()
        >>> ast = parser.parse("def bell(q0: qubit, q1: qubit) { ... }")
    """
    
    def __init__(self) -> None:
        self.tokens: list[Token] = []
        self.pos = 0
    
    def parse(self, source: str) -> dict[str, Any]:
        """Parse Silq source code."""
        lexer = Lexer(source)
        self.tokens = list(lexer.tokenize())
        self.pos = 0
        
        ast = {
            "type": "program",
            "functions": [],
            "source": source,
        }
        
        while not self._is_at_end():
            if self._check(TokenType.DEF):
                ast["functions"].append(self._parse_function())
            else:
                self._advance()
        
        return ast
    
    def _parse_function(self) -> dict[str, Any]:
        """Parse a function definition."""
        self._consume(TokenType.DEF, "Expected 'def'")
        name = self._consume(TokenType.IDENTIFIER, "Expected function name")
        
        self._consume(TokenType.LPAREN, "Expected '('")
        params = self._parse_parameters()
        self._consume(TokenType.RPAREN, "Expected ')'")
        
        # Optional return type
        return_type = None
        if self._check(TokenType.ARROW):
            self._advance()
            return_type = self._consume(TokenType.IDENTIFIER, "Expected return type").value
        
        self._consume(TokenType.LBRACE, "Expected '{'")
        body = self._parse_block()
        self._consume(TokenType.RBRACE, "Expected '}'")
        
        return {
            "type": "function",
            "name": name.value,
            "parameters": params,
            "return_type": return_type,
            "body": body,
        }
    
    def _parse_parameters(self) -> list[dict[str, Any]]:
        """Parse function parameters."""
        params = []
        
        while not self._check(TokenType.RPAREN) and not self._is_at_end():
            name = self._consume(TokenType.IDENTIFIER, "Expected parameter name")
            self._consume(TokenType.COLON, "Expected ':'")
            param_type = self._consume(TokenType.IDENTIFIER, "Expected type").value
            
            # Check for array type
            if self._check(TokenType.LBRACKET):
                self._advance()
                self._consume(TokenType.RBRACKET, "Expected ']'")
                param_type += "[]"
            
            params.append({
                "name": name.value,
                "type": param_type,
            })
            
            if self._check(TokenType.COMMA):
                self._advance()
        
        return params
    
    def _parse_block(self) -> list[dict[str, Any]]:
        """Parse a block of statements."""
        statements = []
        
        while not self._check(TokenType.RBRACE) and not self._is_at_end():
            stmt = self._parse_statement()
            if stmt:
                statements.append(stmt)
        
        return statements
    
    def _parse_statement(self) -> Optional[dict[str, Any]]:
        """Parse a single statement."""
        if self._check(TokenType.RETURN):
            return self._parse_return()
        elif self._check(TokenType.FOR):
            return self._parse_for()
        elif self._check(TokenType.IF):
            return self._parse_if()
        elif self._check(TokenType.IDENTIFIER):
            return self._parse_assignment_or_call()
        else:
            self._advance()
            return None
    
    def _parse_return(self) -> dict[str, Any]:
        """Parse return statement."""
        self._advance()  # consume 'return'
        expr = self._parse_expression()
        if self._check(TokenType.SEMICOLON):
            self._advance()
        return {"type": "return", "value": expr}
    
    def _parse_for(self) -> dict[str, Any]:
        """Parse for loop."""
        self._advance()  # consume 'for'
        var = self._consume(TokenType.IDENTIFIER, "Expected loop variable")
        self._consume(TokenType.IN, "Expected 'in'")
        start = self._consume(TokenType.NUMBER, "Expected start value")
        self._consume(TokenType.DOTDOT, "Expected '..'")
        end = self._consume(TokenType.NUMBER, "Expected end value")
        self._consume(TokenType.LBRACE, "Expected '{'")
        body = self._parse_block()
        self._consume(TokenType.RBRACE, "Expected '}'")
        
        return {
            "type": "for",
            "variable": var.value,
            "start": int(start.value),
            "end": int(end.value),
            "body": body,
        }
    
    def _parse_if(self) -> dict[str, Any]:
        """Parse if statement."""
        self._advance()  # consume 'if'
        condition = self._parse_expression()
        self._consume(TokenType.LBRACE, "Expected '{'")
        then_branch = self._parse_block()
        self._consume(TokenType.RBRACE, "Expected '}'")
        
        else_branch = []
        if self._check(TokenType.ELSE):
            self._advance()
            self._consume(TokenType.LBRACE, "Expected '{'")
            else_branch = self._parse_block()
            self._consume(TokenType.RBRACE, "Expected '}'")
        
        return {
            "type": "if",
            "condition": condition,
            "then": then_branch,
            "else": else_branch,
        }
    
    def _parse_assignment_or_call(self) -> dict[str, Any]:
        """Parse assignment or function call."""
        name = self._advance()
        
        if self._check(TokenType.ASSIGN):
            self._advance()
            value = self._parse_expression()
            if self._check(TokenType.SEMICOLON):
                self._advance()
            return {"type": "assignment", "target": name.value, "value": value}
        
        elif self._check(TokenType.LPAREN):
            self._advance()
            args = []
            while not self._check(TokenType.RPAREN) and not self._is_at_end():
                args.append(self._parse_expression())
                if self._check(TokenType.COMMA):
                    self._advance()
            self._consume(TokenType.RPAREN, "Expected ')'")
            if self._check(TokenType.SEMICOLON):
                self._advance()
            return {"type": "call", "function": name.value, "arguments": args}
        
        return {"type": "identifier", "name": name.value}
    
    def _parse_expression(self) -> dict[str, Any]:
        """Parse expression."""
        if self._check(TokenType.IDENTIFIER):
            name = self._advance()
            
            if self._check(TokenType.LPAREN):
                self._advance()
                args = []
                while not self._check(TokenType.RPAREN) and not self._is_at_end():
                    args.append(self._parse_expression())
                    if self._check(TokenType.COMMA):
                        self._advance()
                self._consume(TokenType.RPAREN, "Expected ')'")
                return {"type": "call", "function": name.value, "arguments": args}
            
            return {"type": "identifier", "name": name.value}
        
        elif self._check(TokenType.NUMBER):
            return {"type": "number", "value": int(self._advance().value)}
        
        elif self._check(TokenType.LPAREN):
            self._advance()
            expr = self._parse_expression()
            self._consume(TokenType.RPAREN, "Expected ')'")
            return expr
        
        return {"type": "unknown"}
    
    def _check(self, token_type: TokenType) -> bool:
        """Check if current token is of given type."""
        if self._is_at_end():
            return False
        return self.tokens[self.pos].type == token_type
    
    def _advance(self) -> Token:
        """Advance to next token and return current."""
        if not self._is_at_end():
            self.pos += 1
        return self.tokens[self.pos - 1]
    
    def _consume(self, token_type: TokenType, message: str) -> Token:
        """Consume token of expected type or raise error."""
        if self._check(token_type):
            return self._advance()
        raise SyntaxError(f"{message} at line {self.tokens[self.pos].line}")
    
    def _is_at_end(self) -> bool:
        """Check if at end of tokens."""
        return self.tokens[self.pos].type == TokenType.EOF


class OpenQASMParser(Parser):
    """
    Parser for OpenQASM quantum assembly language.
    
    Example:
        >>> parser = OpenQASMParser()
        >>> ast = parser.parse("OPENQASM 2.0; qreg q[2]; h q[0];")
    """
    
    def parse(self, source: str) -> dict[str, Any]:
        """Parse OpenQASM source code."""
        ast = {
            "type": "openqasm",
            "version": None,
            "includes": [],
            "qregs": [],
            "cregs": [],
            "gates": [],
            "source": source,
        }
        
        lines = source.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('//'):
                continue
            
            # Version
            if line.startswith('OPENQASM'):
                match = re.match(r'OPENQASM\s+([\d.]+)', line)
                if match:
                    ast["version"] = match.group(1)
            
            # Include
            elif line.startswith('include'):
                match = re.match(r'include\s+"([^"]+)"', line)
                if match:
                    ast["includes"].append(match.group(1))
            
            # Quantum register
            elif line.startswith('qreg'):
                match = re.match(r'qreg\s+(\w+)\s*\[(\d+)\]', line)
                if match:
                    ast["qregs"].append({
                        "name": match.group(1),
                        "size": int(match.group(2)),
                    })
            
            # Classical register
            elif line.startswith('creg'):
                match = re.match(r'creg\s+(\w+)\s*\[(\d+)\]', line)
                if match:
                    ast["cregs"].append({
                        "name": match.group(1),
                        "size": int(match.group(2)),
                    })
            
            # Gate application
            else:
                # Match patterns like "h q[0];" or "cx q[0], q[1];"
                match = re.match(r'(\w+)\s+([^;]+);?', line)
                if match:
                    gate_name = match.group(1)
                    args_str = match.group(2)
                    
                    # Parse qubit arguments
                    qubits = re.findall(r'(\w+)\[(\d+)\]', args_str)
                    
                    if qubits:
                        ast["gates"].append({
                            "name": gate_name.upper(),
                            "qubits": [{"reg": q[0], "index": int(q[1])} for q in qubits],
                        })
        
        return ast


def parse_silq(source: str) -> dict[str, Any]:
    """Convenience function to parse Silq code."""
    return SilqParser().parse(source)


def parse_openqasm(source: str) -> dict[str, Any]:
    """Convenience function to parse OpenQASM code."""
    return OpenQASMParser().parse(source)


def detect_language(source: str) -> str:
    """Detect the quantum programming language from source code."""
    source_lower = source.lower().strip()
    
    if source_lower.startswith('openqasm'):
        return 'openqasm'
    elif 'qreg' in source_lower or 'creg' in source_lower:
        return 'openqasm'
    elif 'def ' in source_lower or ': qubit' in source_lower:
        return 'silq'
    else:
        return 'unknown'


def parse_auto(source: str) -> dict[str, Any]:
    """Automatically detect language and parse."""
    lang = detect_language(source)
    
    if lang == 'silq':
        return parse_silq(source)
    elif lang == 'openqasm':
        return parse_openqasm(source)
    else:
        raise ValueError(f"Cannot detect quantum programming language")
