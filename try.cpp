#include <bits/stdc++.h>
using namespace std;

// ==================== LEXER CLASS ====================
class Lexer {
private:
    string source;
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
        string value;
        int line;
        int column;
        
        Token(TokenType t, const string& v, int l, int c)
            : type(t), value(v), line(l), column(c) {}
        
        Token(TokenType t, int l, int c)
            : type(t), value(""), line(l), column(c) {}
    };

    Lexer(const string& src) 
        : source(src), position(0), line(1), column(1), 
          source_length(src.length()) {}
    
    vector<Token> tokenize() {
        vector<Token> tokens;
        
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
                
                string text = source.substr(start, position - start);
                
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
                
                string num = source.substr(start, position - start);
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
                        throw runtime_error("Unexpected character '&' at line " + 
                                               to_string(line) + ", column " + 
                                               to_string(start_col));
                    }
                    break;
                    
                case '|':
                    advance();
                    if (match('|')) {
                        tokens.push_back(Token(TOK_OR, "||", line, start_col));
                    } else {
                        throw runtime_error("Unexpected character '|' at line " + 
                                               to_string(line) + ", column " + 
                                               to_string(start_col));
                    }
                    break;
                    
                default:
                    throw runtime_error("Unexpected character '" + string(1, c) + 
                                           "' at line " + to_string(line) + 
                                           ", column " + to_string(start_col));
            }
        }
        
        tokens.push_back(Token(TOK_EOF, line, column));
        return tokens;
    }
    
    // Helper function to convert token type to string
    static string tokenTypeToString(TokenType type) {
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

using Token = Lexer::Token;
static inline string tokName(const Token &t) {
    return Lexer::tokenTypeToString(t.type);
}

// ---------- AST ----------
struct AST {
    struct Node { virtual ~Node() = default; };
    using NodePtr = unique_ptr<Node>;

    struct Expr : Node {};
    struct Stmt : Node {};

    struct Number : Expr {
        int value;
        Number(int v) : value(v) {}
    };

    struct Variable : Expr {
        string name;
        Variable(string n) : name(move(n)) {}
    };

    struct Binary : Expr {
        string op;
        NodePtr left, right;
        Binary(string o, NodePtr l, NodePtr r)
            : op(move(o)), left(move(l)), right(move(r)) {}
    };

    struct VarDecl : Stmt {
        string name;
        NodePtr init;
        VarDecl(string n, NodePtr v) : name(move(n)), init(move(v)) {}
    };

    struct Assign : Stmt {
        string name;
        NodePtr expr;
        Assign(string n, NodePtr e) : name(move(n)), expr(move(e)) {}
    };

    struct IfStmt : Stmt {
        NodePtr cond;
        vector<NodePtr> thenBody;
        NodePtr elifCond;
        vector<NodePtr> elifBody;
        vector<NodePtr> elseBody;
    };

    struct WhileStmt : Stmt {
        NodePtr cond;
        vector<NodePtr> body;
    };

    struct ReturnStmt : Stmt {
        NodePtr expr;
    };

    struct Block : Node {
        vector<NodePtr> stmts;
    };

    struct Function : Node {
        string name;
        vector<NodePtr> body;
    };
};

// ---------- Parser ----------
class Parser {
    const vector<Token> &tokens;
    size_t i = 0;

    const Token &peekToken() const {
        if (i >= tokens.size()) throw runtime_error("Unexpected end of tokens");
        return tokens[i];
    }
    bool atEnd() const { return i >= tokens.size() || tokName(tokens[i]) == "EOF"; }

    bool acceptName(const string &name) {
        if (!atEnd() && tokName(tokens[i]) == name) { ++i; return true; }
        return false;
    }
    void expectName(const string &name) {
        if (!acceptName(name)) {
            stringstream ss;
            ss << "Expected token " << name;
            if (!atEnd()) ss << " but got " << tokName(tokens[i]) << "('" << tokens[i].value << "')";
            throw runtime_error(ss.str());
        }
    }

public:
    Parser(const vector<Token> &toks) : tokens(toks) {}

    unique_ptr<AST::Function> parseProgram() {
        // function main() { ... }
        expectName("FUNCTION");
        expectName("MAIN");
        expectName("LPAREN");
        expectName("RPAREN");
        expectName("LBRACE");

        auto fn = make_unique<AST::Function>();
        fn->name = "main";

        while (!acceptName("RBRACE")) {
            fn->body.push_back(parseStatement());
        }
        return fn;
    }

private:
    AST::NodePtr parseStatement() {
        if (tokName(peekToken()) == "INT") return parseVarDecl();
        if (tokName(peekToken()) == "IDENTIFIER") return parseAssign();
        if (tokName(peekToken()) == "IF") return parseIf();
        if (tokName(peekToken()) == "WHILE") return parseWhile();
        if (tokName(peekToken()) == "RETURN") return parseReturn();
        stringstream ss;
        ss << "Unknown statement start: " << tokName(peekToken()) << "('" << peekToken().value << "')";
        throw runtime_error(ss.str());
    }

    AST::NodePtr parseVarDecl() {
        expectName("INT");
        if (tokName(peekToken()) != "IDENTIFIER") throw runtime_error("Expected identifier after int");
        string name = peekToken().value; ++i;
        expectName("ASSIGN");
        auto init = parseExpression();
        expectName("SEMICOLON");
        return make_unique<AST::VarDecl>(name, move(init));
    }

    AST::NodePtr parseAssign() {
        string name = peekToken().value; ++i;
        expectName("ASSIGN");
        auto expr = parseExpression();
        expectName("SEMICOLON");
        return make_unique<AST::Assign>(name, move(expr));
    }

    AST::NodePtr parseIf() {
        expectName("IF");
        expectName("LPAREN");
        auto cond = parseExpression();
        expectName("RPAREN");
        expectName("LBRACE");
        vector<AST::NodePtr> thenBody;
        while (!acceptName("RBRACE")) thenBody.push_back(parseStatement());

        auto node = make_unique<AST::IfStmt>();
        node->cond = move(cond);
        node->thenBody = move(thenBody);

        // optional elif
        if (!atEnd() && tokName(peekToken()) == "ELIF") {
            expectName("ELIF");
            expectName("LPAREN");
            node->elifCond = parseExpression();
            expectName("RPAREN");
            expectName("LBRACE");
            while (!acceptName("RBRACE")) node->elifBody.push_back(parseStatement());
        }

        // optional else
        if (!atEnd() && tokName(peekToken()) == "ELSE") {
            expectName("ELSE");
            expectName("LBRACE");
            while (!acceptName("RBRACE")) node->elseBody.push_back(parseStatement());
        }

        return node;
    }

    AST::NodePtr parseWhile() {
        expectName("WHILE");
        expectName("LPAREN");
        auto cond = parseExpression();
        expectName("RPAREN");
        expectName("LBRACE");
        auto w = make_unique<AST::WhileStmt>();
        w->cond = move(cond);
        while (!acceptName("RBRACE")) w->body.push_back(parseStatement());
        return w;
    }

    AST::NodePtr parseReturn() {
        expectName("RETURN");
        auto expr = parseExpression();
        expectName("SEMICOLON");
        auto r = make_unique<AST::ReturnStmt>();
        r->expr = move(expr);
        return r;
    }

    // Expression parsing with precedence
    // logical_and -> equality ('AND' equality)*
    // equality -> rel (('EQ') rel)*
    // rel -> add (('<'|'>') add)*
    // add -> mul (('+'|'-') mul)*
    // mul -> primary (('*'|'/') primary)*
    // primary -> NUMBER | IDENTIFIER | LPAREN expr RPAREN

    AST::NodePtr parseExpression() { return parseLogicalAnd(); }

    AST::NodePtr parseLogicalAnd() {
        auto left = parseEquality();
        while (!atEnd() && tokName(peekToken()) == "AND") {
            string op = "&&";
            ++i;
            auto right = parseEquality();
            left = make_unique<AST::Binary>(op, move(left), move(right));
        }
        return left;
    }

    AST::NodePtr parseEquality() {
        auto left = parseRel();
        while (!atEnd() && tokName(peekToken()) == "EQ") {
            string op = "==";
            ++i;
            auto right = parseRel();
            left = make_unique<AST::Binary>(op, move(left), move(right));
        }
        return left;
    }

    AST::NodePtr parseRel() {
        auto left = parseAdd();
        while (!atEnd() && (tokName(peekToken()) == "LT" || tokName(peekToken()) == "GT")) {
            string op = tokName(peekToken()) == "LT" ? "<" : ">";
            ++i;
            auto right = parseAdd();
            left = make_unique<AST::Binary>(op, move(left), move(right));
        }
        return left;
    }

    AST::NodePtr parseAdd() {
        auto left = parseMul();
        while (!atEnd() && (tokName(peekToken()) == "+" || tokName(peekToken()) == "PLUS" || tokName(peekToken()) == "MINUS" || tokName(peekToken()) == "-")) {
            string t = tokName(peekToken());
            string op = (t == "MINUS" || t == "-") ? "-" : "+";
            ++i;
            auto right = parseMul();
            left = make_unique<AST::Binary>(op, move(left), move(right));
        }
        return left;
    }

    AST::NodePtr parseMul() {
        auto left = parsePrimary();
        while (!atEnd() && (tokName(peekToken()) == "MUL" || tokName(peekToken()) == "*" || tokName(peekToken()) == "DIV" || tokName(peekToken()) == "/")) {
            string t = tokName(peekToken());
            string op = (t == "DIV" || t == "/") ? "/" : "*";
            ++i;
            auto right = parsePrimary();
            left = make_unique<AST::Binary>(op, move(left), move(right));
        }
        return left;
    }

    AST::NodePtr parsePrimary() {
        if (atEnd()) throw runtime_error("Unexpected end in primary");
        auto &tk = peekToken();
        auto name = tokName(tk);

        if (name == "NUMBER") {
            int val = 0;
            try { val = stoi(tk.value); } catch(...) { val = 0; }
            ++i;
            return make_unique<AST::Number>(val);
        }
        if (name == "IDENTIFIER") {
            string nm = tk.value;
            ++i;
            return make_unique<AST::Variable>(nm);
        }
        if (name == "LPAREN") {
            ++i;
            auto e = parseExpression();
            expectName("RPAREN");
            return e;
        }
        throw runtime_error(string("Invalid primary: ") + name + "('" + tk.value + "')");
    }
};

#include <bits/stdc++.h>
using namespace std;

struct CodeGen {
    vector<string> out;
    vector<string> dataLines;
    int labelCount = 0;

    struct RegisterFile {
        vector<string> regs = {"$t0","$t1","$t2","$t3","$t4","$t5","$t6","$t7"};
        map<string,bool>regUsed;
        unordered_map<string,string> varToReg;
        unordered_map<string,string> regToVar;
        string spillVar = "";

        string alloc(const string &var) {
            if (varToReg.count(var)) return varToReg[var];
            for (auto &r : regs) {
                if (!regToVar.count(r)) {
                    varToReg[var] = r;
                    regToVar[r] = var;
                    return r;
                }
            }
            // spill first register
            string victim = regs[0];
            spillVar = regToVar[victim];
            varToReg.erase(spillVar);
            regToVar.erase(victim);
            varToReg[var] = victim;
            regToVar[victim] = var;
            return victim;
        }

        bool has(const string &var) { return varToReg.count(var); }
        string reg(const string &var) { return varToReg[var]; }
        void clearSpill() { spillVar.clear(); }
    } rf;

    string genLabel(const string &base) { return base + "_" + to_string(labelCount++); }
    void emit(const string &s) { out.push_back(s); }

    void generate(AST::Function &fn) {
        collectVars(fn);

        if (!dataLines.empty()) {
            out.push_back(".data");
            for (auto &d : dataLines) out.push_back(d);
            out.push_back("");
        }

        out.push_back(".text");
        out.push_back(".globl main");
        out.push_back("main:");
        for (auto &stmt : fn.body) genStmt(stmt.get());
        out.push_back("li $v0, 10");
        out.push_back("syscall");
    }

    void writeToFile(const string &filename) {
        ofstream fout(filename);
        for (auto &s : out) fout << s << "\n";
        fout.close();
    }

private:
    void collectVars(AST::Function &fn) {
        unordered_map<string,bool> seen;
        function<void(AST::Node*)> walk = [&](AST::Node* node) {
            if (!node) return;

            if (auto vd = dynamic_cast<AST::VarDecl*>(node)) {
                if (!seen[vd->name]) { dataLines.push_back(vd->name + ": .word 0"); seen[vd->name]=true; }
                walk(vd->init.get());
            } else if (auto as = dynamic_cast<AST::Assign*>(node)) {
                if (!seen[as->name]) { dataLines.push_back(as->name + ": .word 0"); seen[as->name]=true; }
                walk(as->expr.get());
            } else if (auto ifs = dynamic_cast<AST::IfStmt*>(node)) {
                walk(ifs->cond.get());
                for (auto &s : ifs->thenBody) walk(s.get());
                if (ifs->elifCond) walk(ifs->elifCond.get());
                for (auto &s : ifs->elifBody) walk(s.get());
                for (auto &s : ifs->elseBody) walk(s.get());
            } else if (auto w = dynamic_cast<AST::WhileStmt*>(node)) {
                walk(w->cond.get());
                for (auto &s : w->body) walk(s.get());
            } else if (auto r = dynamic_cast<AST::ReturnStmt*>(node)) {
                walk(r->expr.get());
            } else if (auto bin = dynamic_cast<AST::Binary*>(node)) {
                walk(bin->left.get()); walk(bin->right.get());
            } else if (auto v = dynamic_cast<AST::Variable*>(node)) {
                if (!seen[v->name]) { dataLines.push_back(v->name + ": .word 0"); seen[v->name]=true; }
            } else if (auto b = dynamic_cast<AST::Block*>(node)) {
                for (auto &s : b->stmts) walk(s.get());
            }
        };

        for (auto &s : fn.body) walk(s.get());
    }

    string loadVar(const string &var) {
        if (rf.has(var)) return rf.reg(var);
        string r = rf.alloc(var);
        if (!rf.spillVar.empty()) {
            emit("la $at, " + rf.spillVar);
            emit("sw " + r + ", 0($at)");
            rf.clearSpill();
        }
        emit("la $at, " + var);
        emit("lw " + r + ", 0($at)");
        return r;
    }

    void storeVar(const string &var, const string &srcReg) {
        string r = rf.alloc(var);
        if (!rf.spillVar.empty()) {
            emit("la $at, " + rf.spillVar);
            emit("sw " + r + ", 0($at)");
            rf.clearSpill();
        }
        emit("move " + r + ", " + srcReg);
        emit("la $at, " + var);
        emit("sw " + r + ", 0($at)");
    }

    void genStmt(AST::Node *node) {
        if (auto vd = dynamic_cast<AST::VarDecl*>(node)) {
            genExpr(vd->init.get(), "$t9");
            storeVar(vd->name, "$t9");
            return;
        }
        if (auto as = dynamic_cast<AST::Assign*>(node)) {
            genExpr(as->expr.get(), "$t9");
            storeVar(as->name, "$t9");
            return;
        }
        if (auto ifs = dynamic_cast<AST::IfStmt*>(node)) {
            string Lelse = genLabel("Lelse");
            string Lend = genLabel("Lend");

            genCondJumpFalse(ifs->cond.get(), Lelse);
            for (auto &s : ifs->thenBody) genStmt(s.get());
            emit("j " + Lend);

            emit(Lelse + ":");
            if (ifs->elifCond) {
                string LelifFail = genLabel("LelifFail");
                genCondJumpFalse(ifs->elifCond.get(), LelifFail);
                for (auto &s : ifs->elifBody) genStmt(s.get());
                emit("j " + Lend);
                emit(LelifFail + ":");
            }

            for (auto &s : ifs->elseBody) genStmt(s.get());
            emit(Lend + ":");
            return;
        }
        if (auto w = dynamic_cast<AST::WhileStmt*>(node)) {
            string Lstart = genLabel("Lstart");
            string Lend = genLabel("Lend");
            emit(Lstart + ":");
            genCondJumpFalse(w->cond.get(), Lend);
            for (auto &s : w->body) genStmt(s.get());
            emit("j " + Lstart);
            emit(Lend + ":");
            return;
        }
        if (auto r = dynamic_cast<AST::ReturnStmt*>(node)) {
            genExpr(r->expr.get(), "$t9");
            emit("move $v0, $t9");
            emit("li $v0, 1");
            emit("move $a0, $t9");
            emit("syscall");
            emit("li $v0, 10");
            emit("syscall");
            return;
        }
        throw runtime_error("Unhandled statement in CodeGen");
    }

void genExpr(AST::Node *node, const string &dstReg) {
    if (auto num = dynamic_cast<AST::Number*>(node)) {
        emit("li " + dstReg + ", " + to_string(num->value));
        return;
    }

    if (auto var = dynamic_cast<AST::Variable*>(node)) {
        string r = loadVar(var->name); // gets the variable's register
        if (r != dstReg) emit("move " + dstReg + ", " + r);
        return;
    }

    if (auto bin = dynamic_cast<AST::Binary*>(node)) {
        // allocate temporary registers for left and right expressions
        string leftReg = rf.alloc("__tmp_left");
        string rightReg = rf.alloc("__tmp_right");

        genExpr(bin->left.get(), leftReg);
        genExpr(bin->right.get(), rightReg);

        const string &op = bin->op;
        if (op == "+") emit("add " + dstReg + ", " + leftReg + ", " + rightReg);
        else if (op == "-") emit("sub " + dstReg + ", " + leftReg + ", " + rightReg);
        else if (op == "*") emit("mul " + dstReg + ", " + leftReg + ", " + rightReg);
        else if (op == "/") { emit("div " + leftReg + ", " + rightReg); emit("mflo " + dstReg); }
        else if (op == "<") emit("slt " + dstReg + ", " + leftReg + ", " + rightReg);
        else if (op == ">") emit("slt " + dstReg + ", " + rightReg + ", " + leftReg);
        else if (op == "==") {
            string Ltrue = genLabel("Leq_true");
            string Lend = genLabel("Leq_end");
            emit("beq " + leftReg + ", " + rightReg + ", " + Ltrue);
            emit("li " + dstReg + ", 0");
            emit("j " + Lend);
            emit(Ltrue + ":");
            emit("li " + dstReg + ", 1");
            emit(Lend + ":");
        } else if (op == "&&") {
            string Lfalse = genLabel("Land_false");
            string Lend = genLabel("Land_end");
            emit("beq " + leftReg + ", $zero, " + Lfalse);
            emit("beq " + rightReg + ", $zero, " + Lfalse);
            emit("li " + dstReg + ", 1");
            emit("j " + Lend);
            emit(Lfalse + ":");
            emit("li " + dstReg + ", 0");
            emit(Lend + ":");
        }
        return;
    }

    throw runtime_error("Unhandled expr node in codegen");
}

    void genCondJumpFalse(AST::Node *cond, const string &labelFalse) {
        genExpr(cond, "$t9");
        emit("beq $t9, $zero, " + labelFalse);
    }
};



int main() {
    const string input_filename = "file.txt";
    const string output_filename = "output.txt";

    cout << "===========================================\n";
    cout << "Simple Compiler - Lexer+Parser+MIPS\n";
    cout << "Input file: " << input_filename << "\n";
    cout << "Output file: " << output_filename << "\n";
    cout << "===========================================\n\n";

    // Read input file
    ifstream input_file(input_filename);
    if (!input_file.is_open()) {
        cerr << "Error: Cannot open input file '" << input_filename << "'\n";
        cerr << "Please create a file called 'file.txt' with your program.\n";
        return 1;
    }

    stringstream buffer;
    buffer << input_file.rdbuf();
    string source_code = buffer.str();
    input_file.close();

    cout << "Source code read from " << input_filename << ":\n";
    cout << "-------------------------------------------\n";
    cout << source_code;
    cout << "\n-------------------------------------------\n\n";

    try {
        // Create lexer and tokenize
        Lexer lexer(source_code);
        vector<Lexer::Token> tokens = lexer.tokenize();

        cout << "Tokens found (" << tokens.size() << "):\n";
        cout << "-------------------------------------------\n";
        for (const auto& token : tokens) {
            cout << "Line " << token.line << ", Col " << token.column
                      << ": " << Lexer::tokenTypeToString(token.type);
            if (!token.value.empty()) {
                cout << " ('" << token.value << "')";
            }
            cout << "\n";
        }
        cout << "-------------------------------------------\n\n";

        // Parse
        Parser parser(tokens);
        auto func = parser.parseProgram();
        cout << "Parsing finished. Generating MIPS...\n";

        // Generate MIPS
        CodeGen cg;
        cg.generate(*func);
        cg.writeToFile(output_filename);

        cout << "MIPS written to " << output_filename << "\n";
        cout << "Done.\n";
    } catch (const exception &e) {
        cerr << "Compilation error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}

