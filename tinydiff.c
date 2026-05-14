/*
 *  tinydiff.c — single-file reverse-mode automatic differentiation in pure C
 *  Author : AlexseyPip
 *  Version: 1.1
 *  License: MIT
 *  Repo   : github.com/AlexseyPip/tinydiff
 *  Update : 14.05.2026
 *
 *  Tiny compute graph + backpropagation.
 *  No malloc, no dependencies, <512 nodes.
 *  15 primitive ops. Fits in any online compiler.
 */

#include <stdio.h>
#include <math.h>
#include <string.h>

#define TINYDIFF_MAX_VARS 512
#define TINYDIFF_NAME_LEN 128

typedef enum {
    TINYDIFF_OP_INPUT,
    TINYDIFF_OP_CONST,
    TINYDIFF_OP_ADD,
    TINYDIFF_OP_SUB,
    TINYDIFF_OP_MUL,
    TINYDIFF_OP_DIV,
    TINYDIFF_OP_SIN,
    TINYDIFF_OP_COS,
    TINYDIFF_OP_EXP,
    TINYDIFF_OP_LOG,
    TINYDIFF_OP_POW,
    TINYDIFF_OP_NEG,
    TINYDIFF_OP_TANH,
    TINYDIFF_OP_RELU,
    TINYDIFF_OP_SIGMOID
} tinydiff_op;

typedef struct {
    int id;
    char name[TINYDIFF_NAME_LEN];
    double value;
    double grad;
    tinydiff_op op;
    int left;
    int right;
    int visited;
} tinydiff_var;

typedef struct {
    tinydiff_var vars[TINYDIFF_MAX_VARS];
    int count;
} tinydiff_graph;

static tinydiff_graph G;
static int __td_seed = 0;

void tinydiff_init(void) {
    G.count = 0;
    __td_seed = 0;
}

int tinydiff_input(const char *name, double val) {
    tinydiff_var *v = &G.vars[G.count];
    v->id = G.count;
    snprintf(v->name, TINYDIFF_NAME_LEN, "%s", name);
    v->value = val;
    v->grad = 0.0;
    v->op = TINYDIFF_OP_INPUT;
    v->left = -1;
    v->right = -1;
    v->visited = 0;
    return G.count++;
}

int tinydiff_const(double val) {
    char buf[TINYDIFF_NAME_LEN];
    snprintf(buf, TINYDIFF_NAME_LEN, "c%d", __td_seed++);
    tinydiff_var *v = &G.vars[G.count];
    v->id = G.count;
    snprintf(v->name, TINYDIFF_NAME_LEN, "%s", buf);
    v->value = val;
    v->grad = 0.0;
    v->op = TINYDIFF_OP_CONST;
    v->left = -1;
    v->right = -1;
    v->visited = 0;
    return G.count++;
}

static int tinydiff_binary_op(int a, int b, tinydiff_op op) {
    tinydiff_var *va = &G.vars[a];
    tinydiff_var *vb = &G.vars[b];
    tinydiff_var *v = &G.vars[G.count];
    v->id = G.count;
    v->op = op;
    v->left = a;
    v->right = b;
    v->grad = 0.0;
    v->visited = 0;
    v->name[0] = '\0';
    switch (op) {
        case TINYDIFF_OP_ADD: v->value = va->value + vb->value; break;
        case TINYDIFF_OP_SUB: v->value = va->value - vb->value; break;
        case TINYDIFF_OP_MUL: v->value = va->value * vb->value; break;
        case TINYDIFF_OP_DIV: v->value = va->value / vb->value; break;
        default: break;
    }
    return G.count++;
}

static int tinydiff_unary_op(int a, tinydiff_op op) {
    tinydiff_var *va = &G.vars[a];
    tinydiff_var *v = &G.vars[G.count];
    v->id = G.count;
    v->op = op;
    v->left = a;
    v->right = -1;
    v->grad = 0.0;
    v->visited = 0;
    v->name[0] = '\0';
    switch (op) {
        case TINYDIFF_OP_SIN:  v->value = sin(va->value); break;
        case TINYDIFF_OP_COS:  v->value = cos(va->value); break;
        case TINYDIFF_OP_EXP:  v->value = exp(va->value); break;
        case TINYDIFF_OP_LOG:  v->value = log(va->value); break;
        case TINYDIFF_OP_NEG:  v->value = -(va->value); break;
        case TINYDIFF_OP_TANH: v->value = tanh(va->value); break;
        case TINYDIFF_OP_RELU: v->value = va->value > 0.0 ? va->value : 0.0; break;
        case TINYDIFF_OP_SIGMOID: {
            double s = 1.0 / (1.0 + exp(-va->value));
            v->value = s;
            break;
        }
        default: break;
    }
    return G.count++;
}

int tinydiff_add(int a, int b) { return tinydiff_binary_op(a, b, TINYDIFF_OP_ADD); }
int tinydiff_sub(int a, int b) { return tinydiff_binary_op(a, b, TINYDIFF_OP_SUB); }
int tinydiff_mul(int a, int b) { return tinydiff_binary_op(a, b, TINYDIFF_OP_MUL); }
int tinydiff_div(int a, int b) { return tinydiff_binary_op(a, b, TINYDIFF_OP_DIV); }
int tinydiff_sin(int a)      { return tinydiff_unary_op(a, TINYDIFF_OP_SIN); }
int tinydiff_cos(int a)      { return tinydiff_unary_op(a, TINYDIFF_OP_COS); }
int tinydiff_exp(int a)      { return tinydiff_unary_op(a, TINYDIFF_OP_EXP); }
int tinydiff_log(int a)      { return tinydiff_unary_op(a, TINYDIFF_OP_LOG); }
int tinydiff_neg(int a)      { return tinydiff_unary_op(a, TINYDIFF_OP_NEG); }
int tinydiff_tanh(int a)     { return tinydiff_unary_op(a, TINYDIFF_OP_TANH); }
int tinydiff_relu(int a)     { return tinydiff_unary_op(a, TINYDIFF_OP_RELU); }
int tinydiff_sigmoid(int a)  { return tinydiff_unary_op(a, TINYDIFF_OP_SIGMOID); }

int tinydiff_pow(int base, int exp_id) {
    tinydiff_var *vb = &G.vars[base];
    tinydiff_var *ve = &G.vars[exp_id];
    tinydiff_var *v = &G.vars[G.count];
    v->id = G.count;
    v->value = pow(vb->value, ve->value);
    v->grad = 0.0;
    v->op = TINYDIFF_OP_POW;
    v->left = base;
    v->right = exp_id;
    v->visited = 0;
    v->name[0] = '\0';
    return G.count++;
}

void tinydiff_zero_grads(void) {
    for (int i = 0; i < G.count; i++) {
        G.vars[i].grad = 0.0;
    }
}

static void tinydiff_backward_pass(int id) {
    if (id < 0) return;
    tinydiff_var *v = &G.vars[id];

    switch (v->op) {
        case TINYDIFF_OP_INPUT:
        case TINYDIFF_OP_CONST:
            return;
        case TINYDIFF_OP_ADD:
            G.vars[v->left].grad += v->grad;
            G.vars[v->right].grad += v->grad;
            break;
        case TINYDIFF_OP_SUB:
            G.vars[v->left].grad += v->grad;
            G.vars[v->right].grad += -v->grad;
            break;
        case TINYDIFF_OP_MUL:
            G.vars[v->left].grad += G.vars[v->right].value * v->grad;
            G.vars[v->right].grad += G.vars[v->left].value * v->grad;
            break;
        case TINYDIFF_OP_DIV: {
            double rval = G.vars[v->right].value;
            G.vars[v->left].grad += (1.0 / rval) * v->grad;
            G.vars[v->right].grad += (-G.vars[v->left].value / (rval * rval)) * v->grad;
            break;
        }
        case TINYDIFF_OP_SIN:
            G.vars[v->left].grad += cos(G.vars[v->left].value) * v->grad;
            break;
        case TINYDIFF_OP_COS:
            G.vars[v->left].grad += -sin(G.vars[v->left].value) * v->grad;
            break;
        case TINYDIFF_OP_EXP:
            G.vars[v->left].grad += v->value * v->grad;
            break;
        case TINYDIFF_OP_LOG:
            G.vars[v->left].grad += (1.0 / G.vars[v->left].value) * v->grad;
            break;
        case TINYDIFF_OP_POW: {
            double bval = G.vars[v->left].value;
            double eval = G.vars[v->right].value;
            G.vars[v->left].grad += eval * pow(bval, eval - 1.0) * v->grad;
            if (bval > 0.0) {
                G.vars[v->right].grad += v->value * log(bval) * v->grad;
            }
            break;
        }
        case TINYDIFF_OP_NEG:
            G.vars[v->left].grad += -v->grad;
            break;
        case TINYDIFF_OP_TANH:
            G.vars[v->left].grad += (1.0 - v->value * v->value) * v->grad;
            break;
        case TINYDIFF_OP_RELU:
            G.vars[v->left].grad += (G.vars[v->left].value > 0.0 ? 1.0 : 0.0) * v->grad;
            break;
        case TINYDIFF_OP_SIGMOID:
            G.vars[v->left].grad += v->value * (1.0 - v->value) * v->grad;
            break;
    }
}

static int __td_topo[TINYDIFF_MAX_VARS];
static int __td_topo_len;

static void tinydiff_topo_dfs(int id) {
    if (id < 0) return;
    tinydiff_var *v = &G.vars[id];
    if (v->visited) return;
    v->visited = 1;
    tinydiff_topo_dfs(v->left);
    tinydiff_topo_dfs(v->right);
    __td_topo[__td_topo_len++] = id;
}

void tinydiff_backward(int root_id) {
    for (int i = 0; i < G.count; i++) {
        G.vars[i].visited = 0;
    }
    G.vars[root_id].grad = 1.0;
    __td_topo_len = 0;
    tinydiff_topo_dfs(root_id);
    for (int i = __td_topo_len - 1; i >= 0; i--) {
        tinydiff_backward_pass(__td_topo[i]);
    }
}

int tinydiff_expr_string(int id, char *buf, int bufsz) {
    if (id < 0 || id >= G.count) return -1;
    tinydiff_var *v = &G.vars[id];
    switch (v->op) {
        case TINYDIFF_OP_INPUT:
        case TINYDIFF_OP_CONST:
            return snprintf(buf, bufsz, "%s", v->name);
        case TINYDIFF_OP_NEG: {
            char tmp[TINYDIFF_NAME_LEN];
            tinydiff_expr_string(v->left, tmp, sizeof(tmp));
            return snprintf(buf, bufsz, "(-%s)", tmp);
        }
        case TINYDIFF_OP_SIN: case TINYDIFF_OP_COS:
        case TINYDIFF_OP_EXP: case TINYDIFF_OP_LOG:
        case TINYDIFF_OP_TANH: case TINYDIFF_OP_RELU:
        case TINYDIFF_OP_SIGMOID: {
            char tmp[TINYDIFF_NAME_LEN];
            tinydiff_expr_string(v->left, tmp, sizeof(tmp));
            const char *fn = "";
            switch (v->op) {
                case TINYDIFF_OP_SIN: fn = "sin"; break;
                case TINYDIFF_OP_COS: fn = "cos"; break;
                case TINYDIFF_OP_EXP: fn = "exp"; break;
                case TINYDIFF_OP_LOG: fn = "log"; break;
                case TINYDIFF_OP_TANH: fn = "tanh"; break;
                case TINYDIFF_OP_RELU: fn = "relu"; break;
                case TINYDIFF_OP_SIGMOID: fn = "sig"; break;
                default: break;
            }
            return snprintf(buf, bufsz, "%s(%s)", fn, tmp);
        }
        default: {
            char left[TINYDIFF_NAME_LEN], right[TINYDIFF_NAME_LEN];
            tinydiff_expr_string(v->left, left, sizeof(left));
            tinydiff_expr_string(v->right, right, sizeof(right));
            const char *sym = "?";
            switch (v->op) {
                case TINYDIFF_OP_ADD: sym = "+"; break;
                case TINYDIFF_OP_SUB: sym = "-"; break;
                case TINYDIFF_OP_MUL: sym = "*"; break;
                case TINYDIFF_OP_DIV: sym = "/"; break;
                case TINYDIFF_OP_POW: sym = "^"; break;
                default: break;
            }
            return snprintf(buf, bufsz, "(%s%s%s)", left, sym, right);
        }
    }
}

double tinydiff_value(int id) { return G.vars[id].value; }
double tinydiff_grad(int id)  { return G.vars[id].grad; }

void tinydiff_set_input(int id, double val) {
    if (G.vars[id].op == TINYDIFF_OP_INPUT) {
        G.vars[id].value = val;
    }
}

int main(void) {
    tinydiff_init();

    printf("=== TEST 1: z = x*y + sin(x) + exp(y) ===\n");
    int x = tinydiff_input("x", 3.0);
    int y = tinydiff_input("y", 4.0);
    int a = tinydiff_mul(x, y);
    int b = tinydiff_sin(x);
    int c = tinydiff_exp(y);
    int d = tinydiff_add(a, b);
    int z = tinydiff_add(d, c);
    tinydiff_backward(z);
    char buf[TINYDIFF_NAME_LEN];
    tinydiff_expr_string(z, buf, sizeof(buf));
    printf("z = %s = %.6f\n", buf, tinydiff_value(z));
    printf("dz/dx = %.6f (expected: %.6f)\n", tinydiff_grad(x), 4.0 + cos(3.0));
    printf("dz/dy = %.6f (expected: %.6f)\n\n", tinydiff_grad(y), 3.0 + exp(4.0));

    printf("=== TEST 2: zero_grads + re-eval ===\n");
    tinydiff_zero_grads();
    tinydiff_set_input(x, 1.0);
    tinydiff_set_input(y, 2.0);
    tinydiff_backward(z);
    tinydiff_expr_string(z, buf, sizeof(buf));
    printf("z(x=1,y=2) = %s = %.6f\n", buf, tinydiff_value(z));
    printf("dz/dx = %.6f (expected: %.6f)\n", tinydiff_grad(x), 2.0 + cos(1.0));
    printf("dz/dy = %.6f (expected: %.6f)\n\n", tinydiff_grad(y), 1.0 + exp(2.0));

    printf("=== TEST 3: sigmoid + relu ===\n");
    tinydiff_init();
    int x2 = tinydiff_input("x2", 0.5);
    int s = tinydiff_sigmoid(x2);
    int r = tinydiff_relu(s);
    tinydiff_backward(r);
    tinydiff_expr_string(r, buf, sizeof(buf));
    printf("%s = %.6f, grad = %.6f\n\n", buf, tinydiff_value(r), tinydiff_grad(x2));

    printf("=== TEST 4: (a*b + c^d) / e ===\n");
    tinydiff_init();
    int a1 = tinydiff_input("a", 2.0);
    int b1 = tinydiff_input("b", 3.0);
    int c1 = tinydiff_input("c", 4.0);
    int d1 = tinydiff_input("d", 2.0);
    int e1 = tinydiff_input("e", 5.0);
    int ab = tinydiff_mul(a1, b1);
    int cd = tinydiff_pow(c1, d1);
    int num = tinydiff_add(ab, cd);
    int result = tinydiff_div(num, e1);
    tinydiff_backward(result);
    tinydiff_expr_string(result, buf, sizeof(buf));
    printf("result = %s = %.6f\n", buf, tinydiff_value(result));
    printf("da=%.6f db=%.6f dc=%.6f dd=%.6f de=%.6f\n",
           tinydiff_grad(a1), tinydiff_grad(b1),
           tinydiff_grad(c1), tinydiff_grad(d1),
           tinydiff_grad(e1));

    return 0;
}
