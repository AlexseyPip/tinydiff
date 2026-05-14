// tinydiff.c
// Author: AlexseyPip
// Version: 1.0
// Description: 
// Last Update: 14.05.2026

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

typedef struct tinydiff_var {
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

static tinydiff_graph __td_graph;
static int __td_seed = 0;

void tinydiff_init(void) {
    __td_graph.count = 0;
    __td_seed = 0;
}

int tinydiff_input(const char *name, double val) {
    tinydiff_var *v = &__td_graph.vars[__td_graph.count];
    v->id = __td_graph.count;
    snprintf(v->name, TINYDIFF_NAME_LEN, "%s", name);
    v->value = val;
    v->grad = 0.0;
    v->op = TINYDIFF_OP_INPUT;
    v->left = -1;
    v->right = -1;
    v->visited = 0;
    return __td_graph.count++;
}

int tinydiff_const(double val) {
    char buf[TINYDIFF_NAME_LEN];
    snprintf(buf, TINYDIFF_NAME_LEN, "c%d", __td_seed++);
    tinydiff_var *v = &__td_graph.vars[__td_graph.count];
    v->id = __td_graph.count;
    snprintf(v->name, TINYDIFF_NAME_LEN, "%s", buf);
    v->value = val;
    v->grad = 0.0;
    v->op = TINYDIFF_OP_CONST;
    v->left = -1;
    v->right = -1;
    v->visited = 0;
    return __td_graph.count++;
}

int tinydiff_add(int a, int b) {
    tinydiff_var *va = &__td_graph.vars[a];
    tinydiff_var *vb = &__td_graph.vars[b];
    tinydiff_var *v = &__td_graph.vars[__td_graph.count];
    v->id = __td_graph.count;
    v->value = va->value + vb->value;
    v->grad = 0.0;
    v->op = TINYDIFF_OP_ADD;
    v->left = a;
    v->right = b;
    v->visited = 0;
    v->name[0] = '\0';
    return __td_graph.count++;
}

int tinydiff_sub(int a, int b) {
    tinydiff_var *va = &__td_graph.vars[a];
    tinydiff_var *vb = &__td_graph.vars[b];
    tinydiff_var *v = &__td_graph.vars[__td_graph.count];
    v->id = __td_graph.count;
    v->value = va->value - vb->value;
    v->grad = 0.0;
    v->op = TINYDIFF_OP_SUB;
    v->left = a;
    v->right = b;
    v->visited = 0;
    v->name[0] = '\0';
    return __td_graph.count++;
}

int tinydiff_mul(int a, int b) {
    tinydiff_var *va = &__td_graph.vars[a];
    tinydiff_var *vb = &__td_graph.vars[b];
    tinydiff_var *v = &__td_graph.vars[__td_graph.count];
    v->id = __td_graph.count;
    v->value = va->value * vb->value;
    v->grad = 0.0;
    v->op = TINYDIFF_OP_MUL;
    v->left = a;
    v->right = b;
    v->visited = 0;
    v->name[0] = '\0';
    return __td_graph.count++;
}

int tinydiff_div(int a, int b) {
    tinydiff_var *va = &__td_graph.vars[a];
    tinydiff_var *vb = &__td_graph.vars[b];
    tinydiff_var *v = &__td_graph.vars[__td_graph.count];
    v->id = __td_graph.count;
    v->value = va->value / vb->value;
    v->grad = 0.0;
    v->op = TINYDIFF_OP_DIV;
    v->left = a;
    v->right = b;
    v->visited = 0;
    v->name[0] = '\0';
    return __td_graph.count++;
}

int tinydiff_sin(int a) {
    tinydiff_var *va = &__td_graph.vars[a];
    tinydiff_var *v = &__td_graph.vars[__td_graph.count];
    v->id = __td_graph.count;
    v->value = sin(va->value);
    v->grad = 0.0;
    v->op = TINYDIFF_OP_SIN;
    v->left = a;
    v->right = -1;
    v->visited = 0;
    v->name[0] = '\0';
    return __td_graph.count++;
}

int tinydiff_cos(int a) {
    tinydiff_var *va = &__td_graph.vars[a];
    tinydiff_var *v = &__td_graph.vars[__td_graph.count];
    v->id = __td_graph.count;
    v->value = cos(va->value);
    v->grad = 0.0;
    v->op = TINYDIFF_OP_COS;
    v->left = a;
    v->right = -1;
    v->visited = 0;
    v->name[0] = '\0';
    return __td_graph.count++;
}

int tinydiff_exp(int a) {
    tinydiff_var *va = &__td_graph.vars[a];
    tinydiff_var *v = &__td_graph.vars[__td_graph.count];
    v->id = __td_graph.count;
    v->value = exp(va->value);
    v->grad = 0.0;
    v->op = TINYDIFF_OP_EXP;
    v->left = a;
    v->right = -1;
    v->visited = 0;
    v->name[0] = '\0';
    return __td_graph.count++;
}

int tinydiff_log(int a) {
    tinydiff_var *va = &__td_graph.vars[a];
    tinydiff_var *v = &__td_graph.vars[__td_graph.count];
    v->id = __td_graph.count;
    v->value = log(va->value);
    v->grad = 0.0;
    v->op = TINYDIFF_OP_LOG;
    v->left = a;
    v->right = -1;
    v->visited = 0;
    v->name[0] = '\0';
    return __td_graph.count++;
}

int tinydiff_pow(int base, int exp_id) {
    tinydiff_var *vb = &__td_graph.vars[base];
    tinydiff_var *ve = &__td_graph.vars[exp_id];
    tinydiff_var *v = &__td_graph.vars[__td_graph.count];
    v->id = __td_graph.count;
    v->value = pow(vb->value, ve->value);
    v->grad = 0.0;
    v->op = TINYDIFF_OP_POW;
    v->left = base;
    v->right = exp_id;
    v->visited = 0;
    v->name[0] = '\0';
    return __td_graph.count++;
}

int tinydiff_neg(int a) {
    tinydiff_var *va = &__td_graph.vars[a];
    tinydiff_var *v = &__td_graph.vars[__td_graph.count];
    v->id = __td_graph.count;
    v->value = -(va->value);
    v->grad = 0.0;
    v->op = TINYDIFF_OP_NEG;
    v->left = a;
    v->right = -1;
    v->visited = 0;
    v->name[0] = '\0';
    return __td_graph.count++;
}

int tinydiff_tanh(int a) {
    tinydiff_var *va = &__td_graph.vars[a];
    tinydiff_var *v = &__td_graph.vars[__td_graph.count];
    v->id = __td_graph.count;
    v->value = tanh(va->value);
    v->grad = 0.0;
    v->op = TINYDIFF_OP_TANH;
    v->left = a;
    v->right = -1;
    v->visited = 0;
    v->name[0] = '\0';
    return __td_graph.count++;
}

int tinydiff_relu(int a) {
    tinydiff_var *va = &__td_graph.vars[a];
    tinydiff_var *v = &__td_graph.vars[__td_graph.count];
    v->id = __td_graph.count;
    v->value = va->value > 0.0 ? va->value : 0.0;
    v->grad = 0.0;
    v->op = TINYDIFF_OP_RELU;
    v->left = a;
    v->right = -1;
    v->visited = 0;
    v->name[0] = '\0';
    return __td_graph.count++;
}

int tinydiff_sigmoid(int a) {
    tinydiff_var *va = &__td_graph.vars[a];
    tinydiff_var *v = &__td_graph.vars[__td_graph.count];
    v->id = __td_graph.count;
    double s = 1.0 / (1.0 + exp(-va->value));
    v->value = s;
    v->grad = 0.0;
    v->op = TINYDIFF_OP_SIGMOID;
    v->left = a;
    v->right = -1;
    v->visited = 0;
    v->name[0] = '\0';
    return __td_graph.count++;
}

static void tinydiff_clear_visited(void) {
    for (int i = 0; i < __td_graph.count; i++) {
        __td_graph.vars[i].visited = 0;
    }
}

static void tinydiff_backward_pass(int id) {
    if (id < 0) return;
    tinydiff_var *v = &__td_graph.vars[id];

    switch (v->op) {
        case TINYDIFF_OP_INPUT:
        case TINYDIFF_OP_CONST:
            return;

        case TINYDIFF_OP_ADD: {
            tinydiff_var *left = &__td_graph.vars[v->left];
            tinydiff_var *right = &__td_graph.vars[v->right];
            left->grad += v->grad;
            right->grad += v->grad;
            break;
        }
        case TINYDIFF_OP_SUB: {
            tinydiff_var *left = &__td_graph.vars[v->left];
            tinydiff_var *right = &__td_graph.vars[v->right];
            left->grad += v->grad;
            right->grad += -v->grad;
            break;
        }
        case TINYDIFF_OP_MUL: {
            tinydiff_var *left = &__td_graph.vars[v->left];
            tinydiff_var *right = &__td_graph.vars[v->right];
            left->grad += right->value * v->grad;
            right->grad += left->value * v->grad;
            break;
        }
        case TINYDIFF_OP_DIV: {
            tinydiff_var *left = &__td_graph.vars[v->left];
            tinydiff_var *right = &__td_graph.vars[v->right];
            double rval = right->value;
            left->grad += (1.0 / rval) * v->grad;
            right->grad += (-left->value / (rval * rval)) * v->grad;
            break;
        }
        case TINYDIFF_OP_SIN: {
            tinydiff_var *left = &__td_graph.vars[v->left];
            left->grad += cos(left->value) * v->grad;
            break;
        }
        case TINYDIFF_OP_COS: {
            tinydiff_var *left = &__td_graph.vars[v->left];
            left->grad += -sin(left->value) * v->grad;
            break;
        }
        case TINYDIFF_OP_EXP: {
            tinydiff_var *left = &__td_graph.vars[v->left];
            left->grad += v->value * v->grad;
            break;
        }
        case TINYDIFF_OP_LOG: {
            tinydiff_var *left = &__td_graph.vars[v->left];
            left->grad += (1.0 / left->value) * v->grad;
            break;
        }
        case TINYDIFF_OP_POW: {
            tinydiff_var *base = &__td_graph.vars[v->left];
            tinydiff_var *exp_var = &__td_graph.vars[v->right];
            double bval = base->value;
            double eval = exp_var->value;
            base->grad += eval * pow(bval, eval - 1.0) * v->grad;
            if (bval > 0.0) {
                exp_var->grad += v->value * log(bval) * v->grad;
            }
            break;
        }
        case TINYDIFF_OP_NEG: {
            tinydiff_var *left = &__td_graph.vars[v->left];
            left->grad += -v->grad;
            break;
        }
        case TINYDIFF_OP_TANH: {
            tinydiff_var *left = &__td_graph.vars[v->left];
            left->grad += (1.0 - v->value * v->value) * v->grad;
            break;
        }
        case TINYDIFF_OP_RELU: {
            tinydiff_var *left = &__td_graph.vars[v->left];
            left->grad += (left->value > 0.0 ? 1.0 : 0.0) * v->grad;
            break;
        }
        case TINYDIFF_OP_SIGMOID: {
            tinydiff_var *left = &__td_graph.vars[v->left];
            left->grad += v->value * (1.0 - v->value) * v->grad;
            break;
        }
    }
}

static int __td_topo[TINYDIFF_MAX_VARS];
static int __td_topo_len;

static void tinydiff_topo_dfs(int id) {
    if (id < 0) return;
    tinydiff_var *v = &__td_graph.vars[id];
    if (v->visited) return;
    v->visited = 1;
    tinydiff_topo_dfs(v->left);
    tinydiff_topo_dfs(v->right);
    __td_topo[__td_topo_len++] = id;
}

void tinydiff_backward(int root_id) {
    int i;
    for (i = 0; i < __td_graph.count; i++) {
        __td_graph.vars[i].grad = 0.0;
        __td_graph.vars[i].visited = 0;
    }
    __td_graph.vars[root_id].grad = 1.0;

    __td_topo_len = 0;
    tinydiff_topo_dfs(root_id);

    for (i = __td_topo_len - 1; i >= 0; i--) {
        tinydiff_backward_pass(__td_topo[i]);
    }
}

double tinydiff_value(int id) {
    return __td_graph.vars[id].value;
}

double tinydiff_grad(int id) {
    return __td_graph.vars[id].grad;
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
    printf("z = %.6f\n", tinydiff_value(z));
    printf("dz/dx = %.6f (expected: %.6f)\n", tinydiff_grad(x), 4.0 + cos(3.0));
    printf("dz/dy = %.6f (expected: %.6f)\n\n", tinydiff_grad(y), 3.0 + exp(4.0));

    tinydiff_init();
    printf("=== TEST 2: sigmoid(x) then relu ===\n");
    int x2 = tinydiff_input("x2", 0.5);
    int s = tinydiff_sigmoid(x2);
    int r = tinydiff_relu(s);
    tinydiff_backward(r);
    printf("relu(sigmoid(%.1f)) = %.6f\n", 0.5, tinydiff_value(r));
    printf("grad = %.6f\n\n", tinydiff_grad(x2));

    tinydiff_init();
    printf("=== TEST 3: division z = x / y ===\n");
    int x3 = tinydiff_input("x3", 6.0);
    int y3 = tinydiff_input("y3", 2.0);
    int q = tinydiff_div(x3, y3);
    tinydiff_backward(q);
    printf("6/2 = %.6f, dx=%.6f, dy=%.6f\n",
           tinydiff_value(q), tinydiff_grad(x3), tinydiff_grad(y3));

    return 0;
}
