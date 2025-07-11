class Var:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"Var({self.name})"

    def __eq__(self, other):
        return isinstance(other, Var) and self.name == other.name
