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
        vector<NodePtr> elifConds;
        vector<vector<NodePtr>> elifBodies;
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
            while (!atEnd() && tokName(peekToken()) == "ELIF") {
            expectName("ELIF");
            expectName("LPAREN");
            auto elifCond = parseExpression();
            expectName("RPAREN");
            expectName("LBRACE");
            vector<AST::NodePtr> elifBody;
            while (!acceptName("RBRACE")) elifBody.push_back(parseStatement());
            node->elifConds.push_back(move(elifCond));
            node->elifBodies.push_back(move(elifBody));
        }

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

    AST::NodePtr parseExpression() { return parseAssignment(); }

    AST::NodePtr parseAssignment() {
        auto left = parseLogicalOr();
        if (!atEnd() && tokName(peekToken()) == "ASSIGN") { // = 
            ++i;
            auto right = parseAssignment();
            left = make_unique<AST::Binary>("=", move(left), move(right));
        }
        return left;
    }

    AST::NodePtr parseLogicalOr() {
        auto left = parseLogicalAnd();
        while (!atEnd() && tokName(peekToken()) == "OR") { 
            ++i;
            auto right = parseLogicalAnd();
            left = make_unique<AST::Binary>("||", move(left), move(right));
        }
        return left;
    }

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

struct CodeGen {
    vector<string> out;
    vector<string> dataLines;
    int labelCount = 0;

    struct Variable {
        bool isArray;
        int addr;       // memory address (if spilled or array base)
        int size;       // array size
    };

    //────────────────────────────────────────────
    //  REGISTER FILE
    //────────────────────────────────────────────
    struct RegisterFile {

        // Variable registers (spillable)
        vector<string> varRegs = {
            "$s0","$s1","$s2","$s3","$s4","$s5","$s6","$s7"
        };

        // Temporary registers (NOT spillable)
        vector<string> tempRegs = {
            "$t0","$t1","$t2","$t3","$t4","$t5","$t6","$t7"
        };

        unordered_set<string> usedTemp;

        unordered_map<string,string> varToReg;
        unordered_map<string,string> regToVar;

        unordered_map<string,Variable> memory;
        int memAddr = 0;

        //────────────────────────────────────────────
        // Allocate variable register
        //────────────────────────────────────────────
        string alloc(const string &var, CodeGen *cg, bool isArray=false, int arraySize=0) {

            // Already assigned?
            if (varToReg.count(var))
                return varToReg[var];

            // Free register?
            for (auto &r : varRegs) {
                if (!regToVar.count(r)) {
                    assignRegister(var, r, isArray, arraySize);
                    return r;
                }
            }

            string r = varRegs[0];
            string spilledVar = regToVar[r];

            spill(spilledVar, r, cg);

            assignRegister(var, r, isArray, arraySize);
            return r;
        }

        //────────────────────────────────────────────
        // Assign register mapping
        //────────────────────────────────────────────
        void assignRegister(const string &var, const string &r,
                            bool isArray, int arraySize)
        {
            varToReg[var] = r;
            regToVar[r]   = var;

            if (isArray) {
                int base = memAddr;
                memAddr += arraySize * 4;
                memory[var] = {true, base, arraySize};
            } else {
                memory[var] = {false, -1, 1};
            }
        }

        //────────────────────────────────────────────
        // Spill variable from register r
        //────────────────────────────────────────────
        void spill(const string &var, const string &r, CodeGen *cg) {
            Variable &v = memory[var];

            // Arrays already live in memory
            if (!v.isArray) {
                if (v.addr == -1) {
                    v.addr = memAddr;
                    memAddr += 4;
                }
                cg->emit("la $at, " + to_string(v.addr));
                cg->emit("sw " + r + ", 0($at)");
            }

            varToReg.erase(var);
            regToVar.erase(r);
        }

        //────────────────────────────────────────────
        // Allocate temporary register
        //────────────────────────────────────────────
        string allocTemp() {
            for (auto &r : tempRegs) {
                if (!usedTemp.count(r)) {
                    usedTemp.insert(r);
                    return r;
                }
            }
            throw runtime_error("Out of temporary registers");
        }

        void freeTemp(const string &r) {
            usedTemp.erase(r);
        }

        bool has(const string &var) { return varToReg.count(var); }

        string reg(const string &var) { return varToReg[var]; }

    } rf;
    void writeToFile(const string &filename) {
        ofstream fout(filename);
        for (auto &s : out) fout << s << "\n";
        fout.close();
    }
    //────────────────────────────────────────────
    // Helpers
    //────────────────────────────────────────────
    void emit(const string &s) { out.push_back(s); }

    string genLabel(const string &base) {
        return base + "_" + to_string(labelCount++);
    }

    //────────────────────────────────────────────
    // MAIN ENTRY
    //────────────────────────────────────────────
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

        for (auto &s : fn.body) genStmt(s.get());

        out.push_back("syscall");
    }

    //────────────────────────────────────────────
    // Load variable into register
    //────────────────────────────────────────────
    string loadVar(const string &var) {
        if (rf.has(var))
            return rf.reg(var);

        string r = rf.alloc(var, this);
        Variable &v = rf.memory[var];

        // If scalar spilled earlier
        if (!v.isArray && v.addr != -1) {
            emit("la $at, " + to_string(v.addr));
            emit("lw " + r + ", 0($at)");
        }

        return r;
    }

    //────────────────────────────────────────────
    // STORE VARIABLE
    //────────────────────────────────────────────
    void storeVar(const string &var, const string &src) {
        string r = rf.alloc(var, this);
        Variable &v = rf.memory[var];

        if (r != src)
            emit("move " + r + ", " + src);

        if (!v.isArray) {
            if (v.addr == -1) {
                v.addr = rf.memAddr;
                rf.memAddr += 4;
            }
            emit("la $at, " + to_string(v.addr));
            emit("sw " + r + ", 0($at)");
        } else {
            emit("la $at, " + to_string(v.addr));
            emit("sw " + r + ", 0($at)");
        }
    }

    void genStmt(AST::Node *node) {

        if (auto vd = dynamic_cast<AST::VarDecl*>(node)) {
            string r = genExpr(vd->init.get());
            storeVar(vd->name, r);
            rf.freeTemp(r);
            return;
        }

        if (auto as = dynamic_cast<AST::Assign*>(node)) {
            string r = genExpr(as->expr.get());
            storeVar(as->name, r);
            rf.freeTemp(r);
            return;
        }

        if (auto ifs = dynamic_cast<AST::IfStmt*>(node)) {
            string Lelse = genLabel("ELSE");
            string Lend  = genLabel("ENDIF");

            // معالجة شرط 'if' الأول
            genCondJumpFalse(ifs->cond.get(), Lelse);
            for (auto &s : ifs->thenBody) genStmt(s.get());
            emit("j " + Lend);

            // معالجة الشروط الخاصة بـ 'elif'
            for (size_t i = 0; i < ifs->elifConds.size(); ++i) {
                string Lelif = genLabel("ELIF" + to_string(i));  // لكل 'elif' علامة خاصة
                emit(Lelse + ":");
                genCondJumpFalse(ifs->elifConds[i].get(), Lelif);
                for (auto &s : ifs->elifBodies[i]) genStmt(s.get());
                emit("j " + Lend);
                emit(Lelif + ":");
            }

            // معالجة الجسم الخاص بـ 'else'
            emit(Lelse + ":");
            for (auto &s : ifs->elseBody) genStmt(s.get());

            emit(Lend + ":");
            return;
        }

        
        if (auto w = dynamic_cast<AST::WhileStmt*>(node)) {
            string Lstart = genLabel("WHILE");
            string Lend   = genLabel("ENDW");

            emit(Lstart + ":");
            genCondJumpFalse(w->cond.get(), Lend);

            for (auto &s : w->body) genStmt(s.get());
            emit("j " + Lstart);

            emit(Lend + ":");
            return;
        }

        if (auto r = dynamic_cast<AST::ReturnStmt*>(node)) {
            string reg = genExpr(r->expr.get());
            emit("move $v0, " + reg);
            rf.freeTemp(reg);
            emit("syscall");
            return;
        }

        throw runtime_error("Unhandled statement!");
    }


    string genExpr(AST::Node *node) {

        if (auto num = dynamic_cast<AST::Number*>(node)) {
            string r = rf.allocTemp();
            emit("li " + r + ", " + to_string(num->value));
            return r;
        }

        if (auto var = dynamic_cast<AST::Variable*>(node)) {
            return loadVar(var->name);
        }

        if (auto bin = dynamic_cast<AST::Binary*>(node)) {

            string L = genExpr(bin->left.get());
            string R = genExpr(bin->right.get());

            string dst = rf.allocTemp();

            string op = bin->op;

            if (op == "+") emit("add " + dst + ", " + L + ", " + R);
            else if (op == "-") emit("sub " + dst + ", " + L + ", " + R);
            else if (op == "*") emit("mul " + dst + ", " + L + ", " + R);
            else if (op == "/") {
                emit("div " + L + ", " + R);
                emit("mflo " + dst);
            }
            else if (op == "<") emit("slt " + dst + ", " + L + ", " + R);
            else if (op == ">") emit("slt " + dst + ", " + R + ", " + L);
            else if (op == "==") {
                string Lt = genLabel("TRUE");
                string Le = genLabel("END");
                emit("beq " + L + ", " + R + ", " + Lt);
                emit("li " + dst + ", 0");
                emit("j " + Le);
                emit(Lt + ":");
                emit("li " + dst + ", 1");
                emit(Le + ":");
            }

            rf.freeTemp(L);
            rf.freeTemp(R);

            return dst;
        }

        throw runtime_error("Unhandled expression");
    }

    //────────────────────────────────────────────
    // Boolean logic for if/while
    //────────────────────────────────────────────
    void genCondJumpFalse(AST::Node *n, const string &label) {
        string r = genExpr(n);
        emit("beq " + r + ", $zero, " + label);
        rf.freeTemp(r);
    }

    //────────────────────────────────────────────
    // Collect variables into .data
    //────────────────────────────────────────────
    void collectVars(AST::Function &fn) {
        unordered_set<string> seen;

        function<void(AST::Node*)> walk = [&](AST::Node *n) {
            if (!n) return;

            if (auto vd = dynamic_cast<AST::VarDecl*>(n)) {
                if (!seen.count(vd->name)) {
                    dataLines.push_back(vd->name + ": .word 0");
                    seen.insert(vd->name);
                }
                walk(vd->init.get());
            }
            else if (auto as = dynamic_cast<AST::Assign*>(n)) {
                if (!seen.count(as->name)) {
                    dataLines.push_back(as->name + ": .word 0");
                    seen.insert(as->name);
                }
                walk(as->expr.get());
            }
            else if (auto bin = dynamic_cast<AST::Binary*>(n)) {
                walk(bin->left.get());
                walk(bin->right.get());
            }
            else if (auto v = dynamic_cast<AST::Variable*>(n)) {
                if (!seen.count(v->name)) {
                    dataLines.push_back(v->name + ": .word 0");
                    seen.insert(v->name);
                }
            }
        };

        for (auto &s : fn.body) walk(s.get());
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

