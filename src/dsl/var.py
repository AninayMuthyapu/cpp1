class Expression:
    
    def __add__(self, other):
        return ArithmeticExpression('+', self, other)

    def __sub__(self, other):
        return ArithmeticExpression('-', self, other)


class ArithmeticExpression(Expression):
    def __init__(self,op,left,right):
        if op not in ['+', '-']:
            raise ValueError(f"Unsupported operator")
        self.op = op
        self.left = left
        self.right = right

    def __repr__(self):
        return f"ArithmeticExpression({self.op}, {self.left}, {self.right})"
    
    def __eq__(self, other):
        if not isinstance(other, ArithmeticExpression):
            return NotImplemented
        return (self.op == other.op and
                self.left == other.left and
                self.right == other.right)
        

            
class Var(Expression):
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"Var({self.name})"

    def __eq__(self, other):
       
        return isinstance(other, Var) and self.name == other.name

    def __hash__(self):
        
        return hash(self.name)
    

class Conditional(Var):
    conditional_counter=0
    def __init__(self,condition,true_value,false_value,name=None):
        if not isinstance(condition,( Expression,bool)):
            raise TypeError("Condition must be an Expression")
        
        self.true_value = true_value
        self.false_value = false_value
        self.name = name
        if name is None:
            name =f"conditional_expr_{Conditional.conditional_counter}"
            Conditional.conditional_counter += 1

        super().__init__(name)
        self.condition= condition

    def __repr__(self):
        return (f"conditional(if {self.condition} then {repr(self.true_value)} "
                f"else {repr(self.false_value)},name={self.name})")
    def __eq__(self, other):
        if not isinstance(other, Conditional):
            return NotImplemented
        return (self.condition == other.condition and
                self.true_value == other.true_value and
                self.false_value == other.false_value and
                self.name == other.name)
    def __hash__(self):
        return hash((self.condition, self.true_value, self.false_value, self.name))
