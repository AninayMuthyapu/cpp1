
class Expression:
    
    pass

class Comparison(Expression):
    
    def __init__(self, left, op, right):
        self.left = left
        self.op = op
        self.right = right

    def __repr__(self):
        return f"Comparison({self.left}, '{self.op}', {self.right})"

    def __eq__(self, other):
        return (isinstance(other, Comparison) and
                self.op == other.op and
                self.left == other.left and
                self.right == other.right)

class Var(Expression):
    
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"Var('{self.name}')"

    def __eq__(self, other):
        return isinstance(other, Var) and self.name == other.name

    def __hash__(self):
        return hash(self.name)

    
    def __add__(self, other):
        return ArithmeticExpression(self, '+', other)
    
    def __sub__(self, other):
        return ArithmeticExpression(self, '-', other)

    

    def __mul__(self, other):
        return ArithmeticExpression(self, '*', other)

    
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

    def __ne__(self, other):
        return Comparison(self, '!=', other)

class Conditional(Expression):
    
    def __init__(self, condition, true_value, false_value):
        self.condition = condition
        self.true_value = true_value
        self.false_value = false_value

    def __repr__(self):
        return (f"({self.condition} ? {self.true_value} : {self.false_value})")