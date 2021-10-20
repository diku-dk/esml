class Exp:
    domain = set()


    def __add__(self, y):
        return BinOpExp('+', self, y)
    def __radd__(self, x):
        return BinOpExp('+', x, self)
    def __sub__(self, y):
        return BinOpExp('-', self, y)
    def __rsub__(self, x):
        return BinOpExp('-', x, self)
    def __mul__(self, y):
        return BinOpExp('*', self, y)
    def __rmul__(self, x):
        return BinOpExp('*', x, self)
    def __truediv__(self, y):
        return BinOpExp('/', self, y)
    def __rtruediv__(self, x):
        return BinOpExp('/', x, self)
    def __pow__(self, y):
        return BinOpExp('**', self, y)
    def __rpow__(self, x):
        return BinOpExp('**', x, self)
    def __neg__(self):
        return FunExp('neg', [self])
    def __eq__(self, other):
        return Eq(self, other)

    def __repr__(self):
        return str(self)

class BinOpExp(Exp):
    def __init__(self, op, x, y):
        self.op = op
        self.x = x
        self.y = y

    def __str__(self):
        return '(' + str(self.x) + str(self.op) + str(self.y) + ')'

class ConstExp(Exp):
    def __init__(self, v):
        self.v = v

    def __str__(self):
        return str(self.v)

class FieldExp(Exp):
    def __init__(self, f, v):
        self.f = f
        self.v = v

    def __str__(self):
        return str(self.v)

class TimeExp(Exp):
    pass

class DiffExp(Exp):
    def __init__(self, f, xs):
        self.f = f
        self.xs = xs

    def __str__(self):
        return 'D(' + str(self.f) + ',' + ','.join(map(str,self.xs)) + ')'

class FunExp(Exp):
    def __init__(self, f, args):
        assert type(args) == list or type(args) == tuple
        self.f = f
        self.args = args

    def __str__(self):
        return str(self.f) + '(' + ','.join(map(str,self.args)) + ')'

class GradExp(Exp):
    def __init__(self, x):
        assert(isinstance(x, FieldExp))
        self.x = x

    def __eq__(self, other):
        assert len(self.x.f.domain)==len(other)
        return Eq(self, other)

    def __str__(self):
        return 'Gradient(' + str(self.x) + ')'

class Eq:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        return str(self.x) + ' == ' + str(self.y)

    def __repr__(self):
        return str(self)

def MkFunExp(op, arity):
    def f(*es):
        assert len(es) == arity
        return FunExp(op, es)
    return f

Sin = MkFunExp('sin', 1)
Cos = MkFunExp('cos', 1)
Tan = MkFunExp('tan', 1)
Const = ConstExp
Gradient = GradExp

class ExpList:
    def __init__(self, xs):
        self.xs = xs

    def __eq__(self, other):
        assert len(self.xs) == len(other)
        return [Eq(x,y) for (x,y) in zip(self.xs, other)]

def D(f, x, order=1):
    return DiffExp(f, [x for i in range(order)])

class SpatialCoordinate(Exp):
    def __init__(self, v):
        self.v = v

    def __str__(self):
        return self.v

class TimeCoordinate(Exp):
    def __init__(self, v):
        self.v = v

    def __str__(self):
        return self.v

def SpatialCoordinates(*vs):
    return [ SpatialCoordinate(v) for v in vs ]

class FieldHandle:
    def __init__(self, domain):
        self.domain = domain

def Field(coords, vs):
    f = FieldHandle(coords)
    return [ FieldExp(f,v) for v in vs ]
