
class Expression:
    pass

class Comparison(Expression):
    def __init__(self, left, op, right):
        self.left = left
        self.op = op
        self.right = right

    def __repr__(self):
        return f"({self.left} {self.op} {self.right})"

    def __eq__(self, other):
        return (isinstance(other, Comparison) and
                self.op == other.op and
                self.left == other.left and
                self.right == other.right)

    def __hash__(self):
        return hash((self.left, self.op, self.right))

class Var(Expression):
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return self.name

    def __eq__(self, other):
        return isinstance(other, Var) and self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def __add__(self, other):
        return ArithmeticExpression(self, '+', other)
    
    def __sub__(self, other):
        return ArithmeticExpression(self, '-', other)

    def __rsub__(self, other):
        return ArithmeticExpression(other, '-', self)

    def __mul__(self, other):
        return ArithmeticExpression(self, '*', other)
    
    def __truediv__(self, other):
        return ArithmeticExpression(self, '/', other)

    

    def __getitem__(self, index):
        return ArithmeticExpression(self, 'subscript', index)

    def __neg__(self):
        return ArithmeticExpression(self, '-', None)

    def __lt__(self, other):
        return Comparison(self, '<', other)

    def __le__(self, other):
        return Comparison(self, '<=', other)

    def __gt__(self, other):
        return Comparison(self, '>', other)

    def __ge__(self, other):
        return Comparison(self, '>=', other)
        
    def __eq__(self, other):
        return Comparison(self, '==', other)

class ArithmeticExpression(Expression):
    def __init__(self, left, op, right):
        self.left = left
        self.op = op
        self.right = right
    
    def __repr__(self):
        if self.op == 'subscript':
            return f"{self.left}[{self.right}]"
        if self.op == '-':
            if self.right is None:
                return f"-{self.left}"
            return f"({self.left} {self.op} {self.right})"
        return f"({self.left} {self.op} {self.right})"
        
    def __add__(self, other):
        return ArithmeticExpression(self, '+', other)
    
    def __sub__(self, other):
        return ArithmeticExpression(self, '-', other)

    def __mul__(self, other):
        return ArithmeticExpression(self, '*', other)
    
    def __truediv__(self, other):
        return ArithmeticExpression(self, '/', other)


    def __getitem__(self, index):
        return ArithmeticExpression(self, 'subscript', index)
    
    def __lt__(self, other):
        return Comparison(self, '<', other)

    def __le__(self, other):
        return Comparison(self, '<=', other)

    def __gt__(self, other):
        return Comparison(self, '>', other)

    def __ge__(self, other):
        return Comparison(self, '>=', other)

    def __eq__(self, other):
        return Comparison(self, '==', other)

class Conditional(Expression):
    def __init__(self, condition, true_value, false_value):
        self.condition = condition
        self.true_value = true_value
        self.false_value = false_value

    def __repr__(self):
        return (f"({self.condition} ? {self.true_value} : {self.false_value})")

class InverseExperession(Expression):
    def __init__(self, operand_matrix,i,j):
        self.operand_matrix = operand_matrix
        self.i = i
        self.j = j
    def __repr__(self):
        return f"Inverse({self.operand_matrix.name})[{self.i},{self.j}]"

class Summation(Expression):
    def __init__(self, expression, loop_var, start, end):
        self.expression = expression
        self.loop_var = loop_var
        self.start = start
        self.end = end

    def __repr__(self):
        return f"sum({self.loop_var}={self.start}..{self.end}, {self.expression})"

class MatMulExpression(Expression):
    def __init__(self, left_matrix_expr, right_matrix_expr, k_var, inner_dim_size_var):
        self.left_matrix_expr = left_matrix_expr
        self.right_matrix_expr = right_matrix_expr
        self.k_var = k_var
        self.inner_dim_size_var = inner_dim_size_var

    def __repr__(self):
        return f"MatMulExpression(left={self.left_matrix_expr}, right={self.right_matrix_expr})"
