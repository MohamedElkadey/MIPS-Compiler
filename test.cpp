// parser_adt.cpp
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <memory>
#include <map>
#include <stdexcept>
#include <cctype>

// ==================== LEXER (Your existing code) ====================
class Lexer {
private:
    std::string source;
    size_t position;
    size_t line;
    size_t column;
    size_t source_length;
    
    char peek() const {
        if (position >= source_length) return '\0';
        return source[position];
    }
    
    char advance() {
        if (position >= source_length) return '\0';
        char c = source[position++];
        if (c == '\n') {
            line++;
            column = 1;
        } else {
            column++;
        }
        return c;
    }
    
    bool match(char expected) {
        if (peek() == expected) {
            advance();
            return true;
        }
        return false;
    }
    
    void skipWhitespace() {
        while (position < source_length) {
            char c = source[position];
            if (c == ' ' || c == '\t' || c == '\r') {
                advance();
            } else if (c == '\n') {
                advance();
            } else if (c == '#') {  // Skip comments
                while (position < source_length && source[position] != '\n') {
                    advance();
                }
            } else {
                break;
            }
        }
    }
    
    bool isDigit(char c) const { 
        return c >= '0' && c <= '9'; 
    }
    
    bool isAlpha(char c) const { 
        return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_'; 
    }
    
    bool isAlphaNum(char c) const { 
        return isAlpha(c) || isDigit(c); 
    }

public:
    // Token types as enum inside Lexer class
    enum TokenType {
        // Keywords
        TOK_FUNCTION, TOK_MAIN, TOK_IF, TOK_ELIF, TOK_ELSE, 
        TOK_WHILE, TOK_INT, TOK_RETURN,
        
        // Identifiers & literals
        TOK_IDENTIFIER, TOK_NUMBER,
        
        // Operators
        TOK_PLUS, TOK_MINUS, TOK_MUL, TOK_DIV,
        TOK_ASSIGN, TOK_EQ, TOK_NE, TOK_LT, TOK_GT, TOK_LE, TOK_GE,
        TOK_AND, TOK_OR, TOK_NOT,
        
        // Delimiters
        TOK_LPAREN, TOK_RPAREN, TOK_LBRACE, TOK_RBRACE,
        TOK_SEMICOLON,
        
        // Special
        TOK_EOF
    };

    struct Token {
        TokenType type;
        std::string value;
        int line;
        int column;
        
        Token(TokenType t, const std::string& v, int l, int c)
            : type(t), value(v), line(l), column(c) {}
        
        Token(TokenType t, int l, int c)
            : type(t), value(""), line(l), column(c) {}
    };

    Lexer(const std::string& src) 
        : source(src), position(0), line(1), column(1), 
          source_length(src.length()) {}
    
    std::vector<Token> tokenize() {
        std::vector<Token> tokens;
        
        while (position < source_length) {
            skipWhitespace();
            
            if (position >= source_length) break;
            
            char c = peek();
            size_t start_col = column;
            
            // Keywords and identifiers
            if (isAlpha(c)) {
                size_t start = position;
                while (position < source_length && isAlphaNum(peek())) {
                    advance();
                }
                
                std::string text = source.substr(start, position - start);
                
                if (text == "function") {
                    tokens.push_back(Token(TOK_FUNCTION, text, line, start_col));
                } else if (text == "main") {
                    tokens.push_back(Token(TOK_MAIN, text, line, start_col));
                } else if (text == "if") {
                    tokens.push_back(Token(TOK_IF, text, line, start_col));
                } else if (text == "elif") {
                    tokens.push_back(Token(TOK_ELIF, text, line, start_col));
                } else if (text == "else") {
                    tokens.push_back(Token(TOK_ELSE, text, line, start_col));
                } else if (text == "while") {
                    tokens.push_back(Token(TOK_WHILE, text, line, start_col));
                } else if (text == "int") {
                    tokens.push_back(Token(TOK_INT, text, line, start_col));
                } else if (text == "return") {
                    tokens.push_back(Token(TOK_RETURN, text, line, start_col));
                } else {
                    tokens.push_back(Token(TOK_IDENTIFIER, text, line, start_col));
                }
                continue;
            }
            
            // Numbers
            if (isDigit(c)) {
                size_t start = position;
                while (position < source_length && isDigit(peek())) {
                    advance();
                }
                
                std::string num = source.substr(start, position - start);
                tokens.push_back(Token(TOK_NUMBER, num, line, start_col));
                continue;
            }
            
            // Operators and delimiters
            switch (c) {
                case '+': 
                    advance(); 
                    tokens.push_back(Token(TOK_PLUS, "+", line, start_col)); 
                    break;
                case '-': 
                    advance(); 
                    tokens.push_back(Token(TOK_MINUS, "-", line, start_col)); 
                    break;
                case '*': 
                    advance(); 
                    tokens.push_back(Token(TOK_MUL, "*", line, start_col)); 
                    break;
                case '/': 
                    advance(); 
                    tokens.push_back(Token(TOK_DIV, "/", line, start_col)); 
                    break;
                case '(': 
                    advance(); 
                    tokens.push_back(Token(TOK_LPAREN, "(", line, start_col)); 
                    break;
                case ')': 
                    advance(); 
                    tokens.push_back(Token(TOK_RPAREN, ")", line, start_col)); 
                    break;
                case '{': 
                    advance(); 
                    tokens.push_back(Token(TOK_LBRACE, "{", line, start_col)); 
                    break;
                case '}': 
                    advance(); 
                    tokens.push_back(Token(TOK_RBRACE, "}", line, start_col)); 
                    break;
                case ';': 
                    advance(); 
                    tokens.push_back(Token(TOK_SEMICOLON, ";", line, start_col)); 
                    break;
                    
                case '=':
                    advance();
                    if (match('=')) {
                        tokens.push_back(Token(TOK_EQ, "==", line, start_col));
                    } else {
                        tokens.push_back(Token(TOK_ASSIGN, "=", line, start_col));
                    }
                    break;
                    
                case '!':
                    advance();
                    if (match('=')) {
                        tokens.push_back(Token(TOK_NE, "!=", line, start_col));
                    } else {
                        tokens.push_back(Token(TOK_NOT, "!", line, start_col));
                    }
                    break;
                    
                case '<':
                    advance();
                    if (match('=')) {
                        tokens.push_back(Token(TOK_LE, "<=", line, start_col));
                    } else {
                        tokens.push_back(Token(TOK_LT, "<", line, start_col));
                    }
                    break;
                    
                case '>':
                    advance();
                    if (match('=')) {
                        tokens.push_back(Token(TOK_GE, ">=", line, start_col));
                    } else {
                        tokens.push_back(Token(TOK_GT, ">", line, start_col));
                    }
                    break;
                    
                case '&':
                    advance();
                    if (match('&')) {
                        tokens.push_back(Token(TOK_AND, "&&", line, start_col));
                    } else {
                        throw std::runtime_error("Unexpected character '&' at line " + 
                                               std::to_string(line) + ", column " + 
                                               std::to_string(start_col));
                    }
                    break;
                    
                case '|':
                    advance();
                    if (match('|')) {
                        tokens.push_back(Token(TOK_OR, "||", line, start_col));
                    } else {
                        throw std::runtime_error("Unexpected character '|' at line " + 
                                               std::to_string(line) + ", column " + 
                                               std::to_string(start_col));
                    }
                    break;
                    
                default:
                    throw std::runtime_error("Unexpected character '" + std::string(1, c) + 
                                           "' at line " + std::to_string(line) + 
                                           ", column " + std::to_string(start_col));
            }
        }
        
        tokens.push_back(Token(TOK_EOF, line, column));
        return tokens;
    }
    
    // Helper function to convert token type to string
    static std::string tokenTypeToString(TokenType type) {
        switch (type) {
            case TOK_FUNCTION: return "FUNCTION";
            case TOK_MAIN: return "MAIN";
            case TOK_IF: return "IF";
            case TOK_ELIF: return "ELIF";
            case TOK_ELSE: return "ELSE";
            case TOK_WHILE: return "WHILE";
            case TOK_INT: return "INT";
            case TOK_RETURN: return "RETURN";
            case TOK_IDENTIFIER: return "IDENTIFIER";
            case TOK_NUMBER: return "NUMBER";
            case TOK_PLUS: return "PLUS";
            case TOK_MINUS: return "MINUS";
            case TOK_MUL: return "MUL";
            case TOK_DIV: return "DIV";
            case TOK_ASSIGN: return "ASSIGN";
            case TOK_EQ: return "EQ";
            case TOK_NE: return "NE";
            case TOK_LT: return "LT";
            case TOK_GT: return "GT";
            case TOK_LE: return "LE";
            case TOK_GE: return "GE";
            case TOK_AND: return "AND";
            case TOK_OR: return "OR";
            case TOK_NOT: return "NOT";
            case TOK_LPAREN: return "LPAREN";
            case TOK_RPAREN: return "RPAREN";
            case TOK_LBRACE: return "LBRACE";
            case TOK_RBRACE: return "RBRACE";
            case TOK_SEMICOLON: return "SEMICOLON";
            case TOK_EOF: return "EOF";
            default: return "UNKNOWN";
        }
    }
};

// ==================== AST NODES ====================
class ASTNode {
public:
    virtual ~ASTNode() = default;
    virtual void print(int indent = 0) const = 0;
    virtual std::string nodeType() const = 0;
};

class ProgramNode : public ASTNode {
public:
    std::vector<std::unique_ptr<ASTNode>> functions;
    
    void print(int indent = 0) const override {
        std::string spaces(indent, ' ');
        std::cout << spaces << "ProgramNode (" << functions.size() << " functions)\n";
        for (const auto& func : functions) {
            func->print(indent + 2);
        }
    }
    
    std::string nodeType() const override { return "Program"; }
};

class FunctionNode : public ASTNode {
public:
    std::string name;
    std::unique_ptr<class BlockNode> body;
    
    void print(int indent = 0) const override {
        std::string spaces(indent, ' ');
        std::cout << spaces << "FunctionNode: " << name << "\n";
        if (body) {
            body->print(indent + 2);
        }
    }
    
    std::string nodeType() const override { return "Function"; }
};

class BlockNode : public ASTNode {
public:
    std::vector<std::unique_ptr<ASTNode>> statements;
    
    void print(int indent = 0) const override {
        std::string spaces(indent, ' ');
        std::cout << spaces << "BlockNode (" << statements.size() << " statements)\n";
        for (const auto& stmt : statements) {
            stmt->print(indent + 2);
        }
    }
    
    std::string nodeType() const override { return "Block"; }
};

class VarDeclNode : public ASTNode {
public:
    std::string type;
    std::string name;
    std::unique_ptr<ASTNode> initializer;
    
    void print(int indent = 0) const override {
        std::string spaces(indent, ' ');
        std::cout << spaces << "VarDeclNode: " << type << " " << name;
        if (initializer) {
            std::cout << " = ";
            initializer->print(0);
        } else {
            std::cout << "\n";
        }
    }
    
    std::string nodeType() const override { return "VarDecl"; }
};

class AssignNode : public ASTNode {
public:
    std::string name;
    std::unique_ptr<ASTNode> value;
    
    void print(int indent = 0) const override {
        std::string spaces(indent, ' ');
        std::cout << spaces << "AssignNode: " << name << " = ";
        value->print(0);
    }
    
    std::string nodeType() const override { return "Assign"; }
};

class IfNode : public ASTNode {
public:
    std::unique_ptr<ASTNode> condition;
    std::unique_ptr<BlockNode> thenBlock;
    std::vector<std::pair<std::unique_ptr<ASTNode>, std::unique_ptr<BlockNode>>> elifBlocks;
    std::unique_ptr<BlockNode> elseBlock;
    
    void print(int indent = 0) const override {
        std::string spaces(indent, ' ');
        std::cout << spaces << "IfNode:\n";
        
        std::cout << spaces << "  Condition: ";
        condition->print(0);
        
        std::cout << spaces << "  Then:\n";
        if (thenBlock) thenBlock->print(indent + 4);
        
        for (size_t i = 0; i < elifBlocks.size(); i++) {
            std::cout << spaces << "  Elif " << (i + 1) << ":\n";
            std::cout << spaces << "    Condition: ";
            elifBlocks[i].first->print(0);
            elifBlocks[i].second->print(indent + 6);
        }
        
        if (elseBlock) {
            std::cout << spaces << "  Else:\n";
            elseBlock->print(indent + 4);
        }
    }
    
    std::string nodeType() const override { return "If"; }
};

class WhileNode : public ASTNode {
public:
    std::unique_ptr<ASTNode> condition;
    std::unique_ptr<BlockNode> body;
    
    void print(int indent = 0) const override {
        std::string spaces(indent, ' ');
        std::cout << spaces << "WhileNode:\n";
        std::cout << spaces << "  Condition: ";
        condition->print(0);
        std::cout << spaces << "  Body:\n";
        if (body) body->print(indent + 4);
    }
    
    std::string nodeType() const override { return "While"; }
};

class ReturnNode : public ASTNode {
public:
    std::unique_ptr<ASTNode> value;
    
    void print(int indent = 0) const override {
        std::string spaces(indent, ' ');
        std::cout << spaces << "ReturnNode: ";
        if (value) {
            value->print(0);
        } else {
            std::cout << "void\n";
        }
    }
    
    std::string nodeType() const override { return "Return"; }
};

class BinaryOpNode : public ASTNode {
public:
    Lexer::Token op;
    std::unique_ptr<ASTNode> left;
    std::unique_ptr<ASTNode> right;
    
    void print(int indent = 0) const override {
        std::string spaces(indent, ' ');
        std::cout << spaces << "BinaryOpNode: " << op.value << "\n";
        std::cout << spaces << "  Left: ";
        left->print(0);
        std::cout << spaces << "  Right: ";
        right->print(0);
    }
    
    std::string nodeType() const override { return "BinaryOp"; }
};

class IdentifierNode : public ASTNode {
public:
    std::string name;
    
    void print(int indent = 0) const override {
        std::string spaces(indent, ' ');
        std::cout << spaces << "IdentifierNode: " << name << "\n";
    }
    
    std::string nodeType() const override { return "Identifier"; }
};

class NumberNode : public ASTNode {
public:
    int value;
    
    void print(int indent = 0) const override {
        std::string spaces(indent, ' ');
        std::cout << spaces << "NumberNode: " << value << "\n";
    }
    
    std::string nodeType() const override { return "Number"; }
};

// ==================== PARSER ====================
class Parser {
private:
    std::vector<Lexer::Token> tokens;
    size_t current;
    
    const Lexer::Token& peek() const {
        return tokens[current];
    }
    
    const Lexer::Token& advance() {
        if (!isAtEnd()) {
            return tokens[current++];
        }
        return tokens.back(); // Return EOF token
    }
    
    bool check(Lexer::TokenType type) const {
        if (isAtEnd()) return false;
        return peek().type == type;
    }
    
    bool match(Lexer::TokenType type) {
        if (check(type)) {
            advance();
            return true;
        }
        return false;
    }
    
    Lexer::Token consume(Lexer::TokenType type, const std::string& message) {
        if (check(type)) {
            return advance();
        }
        throw std::runtime_error(message + " at line " + std::to_string(peek().line));
    }
    
    bool isAtEnd() const {
        return peek().type == Lexer::TOK_EOF;
    }
    
    void printCurrentToken() const {
        if (!isAtEnd()) {
            const auto& token = peek();
            std::cout << "Current token: " << Lexer::tokenTypeToString(token.type);
            if (!token.value.empty()) {
                std::cout << " ('" << token.value << "')";
            }
            std::cout << " at line " << token.line << "\n";
        }
    }

public:
    Parser(std::vector<Lexer::Token> tokens) : tokens(std::move(tokens)), current(0) {}
    
    std::unique_ptr<ProgramNode> parse() {
        auto program = std::make_unique<ProgramNode>();
        
        while (!isAtEnd()) {
            if (match(Lexer::TOK_FUNCTION)) {
                program->functions.push_back(parseFunction());
            } else {
                // Skip unexpected tokens for error recovery
                std::cerr << "Warning: Expected function declaration, got " 
                         << Lexer::tokenTypeToString(peek().type) << " at line " 
                         << peek().line << ". Skipping.\n";
                advance();
            }
        }
        
        return program;
    }
    
    void printTokens() const {
        std::cout << "\n=== TOKEN LIST ===\n";
        for (size_t i = 0; i < tokens.size(); i++) {
            const auto& token = tokens[i];
            std::cout << "[" << i << "] ";
            if (i == current) std::cout << "-> ";
            else std::cout << "   ";
            
            std::cout << Lexer::tokenTypeToString(token.type);
            if (!token.value.empty()) {
                std::cout << " ('" << token.value << "')";
            }
            std::cout << " at line " << token.line << ", col " << token.column << "\n";
        }
        std::cout << "==================\n\n";
    }

private:
    std::unique_ptr<FunctionNode> parseFunction() {
        auto func = std::make_unique<FunctionNode>();
        
        // Function name
        if (check(Lexer::TOK_MAIN)) {
            // Handle "main" as a keyword
            auto token = advance();
            func->name = token.value;
        } else {
            auto nameToken = consume(Lexer::TOK_IDENTIFIER, "Expected function name");
            func->name = nameToken.value;
        }
        
        // Parameters
        consume(Lexer::TOK_LPAREN, "Expected '(' after function name");
        consume(Lexer::TOK_RPAREN, "Expected ')'");
        
        // Function body
        func->body = parseBlock();
        
        return func;
    }
    
    std::unique_ptr<BlockNode> parseBlock() {
        consume(Lexer::TOK_LBRACE, "Expected '{' to start block");
        
        auto block = std::make_unique<BlockNode>();
        
        while (!check(Lexer::TOK_RBRACE) && !isAtEnd()) {
            try {
                block->statements.push_back(parseStatement());
            } catch (const std::runtime_error& e) {
                std::cerr << "Error parsing statement: " << e.what() << "\n";
                // Skip to next statement (look for semicolon or brace)
                while (!check(Lexer::TOK_SEMICOLON) && 
                       !check(Lexer::TOK_RBRACE) && 
                       !isAtEnd()) {
                    advance();
                }
                if (check(Lexer::TOK_SEMICOLON)) advance();
            }
        }
        
        consume(Lexer::TOK_RBRACE, "Expected '}' to end block");
        return block;
    }
    
    std::unique_ptr<ASTNode> parseStatement() {
        if (check(Lexer::TOK_INT)) {
            return parseVarDecl();
        }
        if (check(Lexer::TOK_IF)) {
            return parseIf();
        }
        if (check(Lexer::TOK_WHILE)) {
            return parseWhile();
        }
        if (check(Lexer::TOK_RETURN)) {
            return parseReturn();
        }
        
        // Check for assignment: identifier followed by '='
        if (check(Lexer::TOK_IDENTIFIER)) {
            // Look ahead to see if next token is '='
            if (current + 1 < tokens.size() && 
                tokens[current + 1].type == Lexer::TOK_ASSIGN) {
                return parseAssign();
            }
        }
        
        throw std::runtime_error("Expected statement, got " + 
                                Lexer::tokenTypeToString(peek().type));
    }
    
    std::unique_ptr<VarDeclNode> parseVarDecl() {
        consume(Lexer::TOK_INT, "Expected 'int'");
        
        auto varDecl = std::make_unique<VarDeclNode>();
        varDecl->type = "int";
        
        auto nameToken = consume(Lexer::TOK_IDENTIFIER, "Expected variable name");
        varDecl->name = nameToken.value;
        
        if (match(Lexer::TOK_ASSIGN)) {
            varDecl->initializer = parseExpression();
        }
        
        consume(Lexer::TOK_SEMICOLON, "Expected ';' after variable declaration");
        return varDecl;
    }
    
    std::unique_ptr<AssignNode> parseAssign() {
        auto assign = std::make_unique<AssignNode>();
        
        auto nameToken = consume(Lexer::TOK_IDENTIFIER, "Expected variable name");
        assign->name = nameToken.value;
        
        consume(Lexer::TOK_ASSIGN, "Expected '=' after variable name");
        assign->value = parseExpression();
        
        consume(Lexer::TOK_SEMICOLON, "Expected ';' after assignment");
        return assign;
    }
    
    std::unique_ptr<IfNode> parseIf() {
        auto ifNode = std::make_unique<IfNode>();
        
        consume(Lexer::TOK_IF, "Expected 'if'");
        consume(Lexer::TOK_LPAREN, "Expected '(' after 'if'");
        ifNode->condition = parseExpression();
        consume(Lexer::TOK_RPAREN, "Expected ')' after condition");
        
        ifNode->thenBlock = parseBlock();
        
        // Parse elif blocks
        while (match(Lexer::TOK_ELIF)) {
            consume(Lexer::TOK_LPAREN, "Expected '(' after 'elif'");
            auto condition = parseExpression();
            consume(Lexer::TOK_RPAREN, "Expected ')' after condition");
            auto block = parseBlock();
            ifNode->elifBlocks.emplace_back(std::move(condition), std::move(block));
        }
        
        // Parse else block
        if (match(Lexer::TOK_ELSE)) {
            ifNode->elseBlock = parseBlock();
        }
        
        return ifNode;
    }
    
    std::unique_ptr<WhileNode> parseWhile() {
        auto whileNode = std::make_unique<WhileNode>();
        
        consume(Lexer::TOK_WHILE, "Expected 'while'");
        consume(Lexer::TOK_LPAREN, "Expected '(' after 'while'");
        whileNode->condition = parseExpression();
        consume(Lexer::TOK_RPAREN, "Expected ')' after condition");
        
        whileNode->body = parseBlock();
        return whileNode;
    }
    
    std::unique_ptr<ReturnNode> parseReturn() {
        auto returnNode = std::make_unique<ReturnNode>();
        
        consume(Lexer::TOK_RETURN, "Expected 'return'");
        
        if (!check(Lexer::TOK_SEMICOLON)) {
            returnNode->value = parseExpression();
        }
        
        consume(Lexer::TOK_SEMICOLON, "Expected ';' after return");
        return returnNode;
    }
    
    std::unique_ptr<ASTNode> parseExpression() {
        return parseLogicalOr();
    }
    
    std::unique_ptr<ASTNode> parseLogicalOr() {
        auto expr = parseLogicalAnd();
        
        while (match(Lexer::TOK_OR)) {
            auto op = tokens[current - 1];
            auto right = parseLogicalAnd();
            
            auto node = std::make_unique<BinaryOpNode>();
            node->op = op;
            node->left = std::move(expr);
            node->right = std::move(right);
            expr = std::move(node);
        }
        
        return expr;
    }
    
    std::unique_ptr<ASTNode> parseLogicalAnd() {
        auto expr = parseEquality();
        
        while (match(Lexer::TOK_AND)) {
            auto op = tokens[current - 1];
            auto right = parseEquality();
            
            auto node = std::make_unique<BinaryOpNode>();
            node->op = op;
            node->left = std::move(expr);
            node->right = std::move(right);
            expr = std::move(node);
        }
        
        return expr;
    }
    
    std::unique_ptr<ASTNode> parseEquality() {
        auto expr = parseComparison();
        
        while (match(Lexer::TOK_EQ) || match(Lexer::TOK_NE)) {
            auto op = tokens[current - 1];
            auto right = parseComparison();
            
            auto node = std::make_unique<BinaryOpNode>();
            node->op = op;
            node->left = std::move(expr);
            node->right = std::move(right);
            expr = std::move(node);
        }
        
        return expr;
    }
    
    std::unique_ptr<ASTNode> parseComparison() {
        auto expr = parseTerm();
        
        while (match(Lexer::TOK_LT) || match(Lexer::TOK_GT) || 
               match(Lexer::TOK_LE) || match(Lexer::TOK_GE)) {
            auto op = tokens[current - 1];
            auto right = parseTerm();
            
            auto node = std::make_unique<BinaryOpNode>();
            node->op = op;
            node->left = std::move(expr);
            node->right = std::move(right);
            expr = std::move(node);
        }
        
        return expr;
    }
    
    std::unique_ptr<ASTNode> parseTerm() {
        auto expr = parseFactor();
        
        while (match(Lexer::TOK_PLUS) || match(Lexer::TOK_MINUS)) {
            auto op = tokens[current - 1];
            auto right = parseFactor();
            
            auto node = std::make_unique<BinaryOpNode>();
            node->op = op;
            node->left = std::move(expr);
            node->right = std::move(right);
            expr = std::move(node);
        }
        
        return expr;
    }
    
    std::unique_ptr<ASTNode> parseFactor() {
        auto expr = parseUnary();
        
        while (match(Lexer::TOK_MUL) || match(Lexer::TOK_DIV)) {
            auto op = tokens[current - 1];
            auto right = parseUnary();
            
            auto node = std::make_unique<BinaryOpNode>();
            node->op = op;
            node->left = std::move(expr);
            node->right = std::move(right);
            expr = std::move(node);
        }
        
        return expr;
    }
    
    std::unique_ptr<ASTNode> parseUnary() {
        if (match(Lexer::TOK_MINUS) || match(Lexer::TOK_NOT)) {
            auto op = tokens[current - 1];
            auto right = parseUnary();
            
            // For negative numbers: -5 becomes 0 - 5
            if (op.type == Lexer::TOK_MINUS) {
                auto zero = std::make_unique<NumberNode>();
                zero->value = 0;
                
                auto node = std::make_unique<BinaryOpNode>();
                node->op = op;
                node->left = std::move(zero);
                node->right = std::move(right);
                return node;
            }
            // For logical NOT: !expr
            else {
                // We'll implement NOT as comparison with 0
                auto zero = std::make_unique<NumberNode>();
                zero->value = 0;
                
                auto node = std::make_unique<BinaryOpNode>();
                node->op = op;
                node->left = std::move(right);
                node->right = std::move(zero);
                return node;
            }
        }
        
        return parsePrimary();
    }
    
    std::unique_ptr<ASTNode> parsePrimary() {
        if (match(Lexer::TOK_NUMBER)) {
            auto num = std::make_unique<NumberNode>();
            try {
                num->value = std::stoi(tokens[current - 1].value);
            } catch (...) {
                throw std::runtime_error("Invalid number: " + tokens[current - 1].value);
            }
            return num;
        }
        
        if (match(Lexer::TOK_IDENTIFIER)) {
            auto ident = std::make_unique<IdentifierNode>();
            ident->name = tokens[current - 1].value;
            return ident;
        }
        
        if (match(Lexer::TOK_LPAREN)) {
            auto expr = parseExpression();
            consume(Lexer::TOK_RPAREN, "Expected ')' after expression");
            return expr;
        }
        
        throw std::runtime_error("Expected expression, got " + 
                                Lexer::tokenTypeToString(peek().type));
    }
};

// ==================== TESTER ====================
class ParserTester {
public:
    static void testWithFile(const std::string& filename) {
        std::cout << "\n" << std::string(60, '=') << "\n";
        std::cout << "TESTING PARSER WITH FILE: " << filename << "\n";
        std::cout << std::string(60, '=') << "\n";
        
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error: Cannot open file " << filename << "\n";
            return;
        }
        
        std::stringstream buffer;
        buffer << file.rdbuf();
        std::string source = buffer.str();
        file.close();
        
        testSource(source);
    }
    
    static void testSource(const std::string& source) {
        std::cout << "\nSOURCE CODE:\n";
        std::cout << std::string(40, '-') << "\n";
        std::cout << source;
        std::cout << std::string(40, '-') << "\n";
        
        try {
            // Step 1: Lexical analysis
            Lexer lexer(source);
            auto tokens = lexer.tokenize();
            std::cout << "\n✅ Lexer: Found " << tokens.size() << " tokens\n";
            
            // Step 2: Parsing
            Parser parser(tokens);
            parser.printTokens();  // Show all tokens
            
            auto program = parser.parse();
            std::cout << "✅ Parser: Successfully built AST\n";
            
            // Step 3: AST Analysis
            std::cout << "\n=== AST ANALYSIS ===\n";
            analyzeAST(program.get());
            
            // Step 4: Print AST
            std::cout << "\n=== AST STRUCTURE ===\n";
            program->print();
            
        } catch (const std::exception& e) {
            std::cerr << "\n❌ ERROR: " << e.what() << "\n";
        }
    }
    
    static void runTestSuite() {
        std::cout << "\n" << std::string(60, '=') << "\n";
        std::cout << "PARSER TEST SUITE\n";
        std::cout << std::string(60, '=') << "\n";
        
        // Test 1: Simple program
        std::cout << "\n[TEST 1] Simple program\n";
        testSource(R"(
function main() {
    int x = 5;
    return x;
}
)");
        
        // Test 2: If-elif-else
        std::cout << "\n\n[TEST 2] If-elif-else\n";
        testSource(R"(
function main() {
    int score = 85;
    
    if (score >= 90) {
        return 1;
    } elif (score >= 80) {
        return 2;
    } elif (score >= 70) {
        return 3;
    } else {
        return 4;
    }
}
)");
        
        // Test 3: While loop
        std::cout << "\n\n[TEST 3] While loop\n";
        testSource(R"(
function main() {
    int i = 0;
    int sum = 0;
    
    while (i < 10) {
        sum = sum + i;
        i = i + 1;
    }
    
    return sum;
}
)");
        
        // Test 4: Complex expressions
        std::cout << "\n\n[TEST 4] Complex expressions\n";
        testSource(R"(
function main() {
    int a = 10;
    int b = 20;
    int c = 30;
    
    if (a < b && b < c || a == 10) {
        return (a + b) * c / 2;
    }
    
    return -1;
}
)");
        
        // Test 5: Nested control structures
        std::cout << "\n\n[TEST 5] Nested control structures\n";
        testSource(R"(
function main() {
    int x = 5;
    
    if (x > 0) {
        while (x < 100) {
            if (x % 2 == 0) {
                x = x * 2;
            } else {
                x = x + 1;
            }
        }
    }
    
    return x;
}
)");
    }
    
private:
    static void analyzeAST(ASTNode* node) {
        std::map<std::string, int> nodeCounts;
        countNodes(node, nodeCounts);
        
        std::cout << "Node statistics:\n";
        for (const auto& [type, count] : nodeCounts) {
            std::cout << "  " << type << ": " << count << "\n";
        }
        
        // Additional validation
        if (auto program = dynamic_cast<ProgramNode*>(node)) {
            std::cout << "\nProgram validation:\n";
            std::cout << "  Functions: " << program->functions.size() << "\n";
            
            bool hasMain = false;
            for (const auto& func : program->functions) {
                if (auto fn = dynamic_cast<FunctionNode*>(func.get())) {
                    if (fn->name == "main") hasMain = true;
                    std::cout << "    - " << fn->name << "\n";
                }
            }
            
            if (hasMain) {
                std::cout << "  ✅ Has main function\n";
            } else {
                std::cout << "  ⚠️ No main function found\n";
            }
        }
    }
    
    static void countNodes(ASTNode* node, std::map<std::string, int>& counts) {
        if (!node) return;
        
        counts[node->nodeType()]++;
        
        // Recursively count child nodes
        if (auto program = dynamic_cast<ProgramNode*>(node)) {
            for (const auto& func : program->functions) {
                countNodes(func.get(), counts);
            }
        }
        else if (auto func = dynamic_cast<FunctionNode*>(node)) {
            if (func->body) countNodes(func->body.get(), counts);
        }
        else if (auto block = dynamic_cast<BlockNode*>(node)) {
            for (const auto& stmt : block->statements) {
                countNodes(stmt.get(), counts);
            }
        }
        else if (auto varDecl = dynamic_cast<VarDeclNode*>(node)) {
            if (varDecl->initializer) countNodes(varDecl->initializer.get(), counts);
        }
        else if (auto assign = dynamic_cast<AssignNode*>(node)) {
            if (assign->value) countNodes(assign->value.get(), counts);
        }
        else if (auto ifNode = dynamic_cast<IfNode*>(node)) {
            if (ifNode->condition) countNodes(ifNode->condition.get(), counts);
            if (ifNode->thenBlock) countNodes(ifNode->thenBlock.get(), counts);
            for (const auto& elif : ifNode->elifBlocks) {
                if (elif.first) countNodes(elif.first.get(), counts);
                if (elif.second) countNodes(elif.second.get(), counts);
            }
            if (ifNode->elseBlock) countNodes(ifNode->elseBlock.get(), counts);
        }
        else if (auto whileNode = dynamic_cast<WhileNode*>(node)) {
            if (whileNode->condition) countNodes(whileNode->condition.get(), counts);
            if (whileNode->body) countNodes(whileNode->body.get(), counts);
        }
        else if (auto ret = dynamic_cast<ReturnNode*>(node)) {
            if (ret->value) countNodes(ret->value.get(), counts);
        }
        else if (auto binOp = dynamic_cast<BinaryOpNode*>(node)) {
            if (binOp->left) countNodes(binOp->left.get(), counts);
            if (binOp->right) countNodes(binOp->right.get(), counts);
        }
    }
};

// ==================== MAIN ====================
int main() {
    std::cout << "Parser & AST Test System\n";
    std::cout << "========================\n\n";
    
    int choice;
    std::cout << "Choose testing mode:\n";
    std::cout << "1. Test with file.txt\n";
    std::cout << "2. Run test suite\n";
    std::cout << "3. Enter custom code\n";
    std::cout << "Choice: ";
    
    std::cin >> choice;
    std::cin.ignore(); // Clear newline
    
    if (choice == 1) {
        // Test with file.txt
        ParserTester::testWithFile("file.txt");
    } 
    else if (choice == 2) {
        // Run test suite
        ParserTester::runTestSuite();
    } 
    else if (choice == 3) {
        // Enter custom code
        std::cout << "\nEnter your code (end with empty line):\n";
        std::string line;
        std::string source;
        
        while (true) {
            std::getline(std::cin, line);
            if (line.empty()) break;
            source += line + "\n";
        }
        
        ParserTester::testSource(source);
    } 
    else {
        std::cerr << "Invalid choice\n";
        return 1;
    }
    
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "TEST COMPLETE\n";
    std::cout << std::string(60, '=') << "\n";
    
    return 0;
}