class Exp:
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
        return FunExp('neg', self)
    def __eq__(self, other):
        return Eq(self, other)

class BinOpExp(Exp):
    def __init__(self, op, x, y):
        self.op = op
        self.x = x
        self.y = y

    def __str__(self):
        return '(' + str(self.x) + self.op + str(self.y) + ')'

class ConstExp(Exp):
    def __init__(self, v):
        self.v = v

    def __str__(self):
        return self.v

class FieldExp(Exp):
    def __init__(self, f, i):
        self.f = f
        self.i = i

class TimeExp(Exp):
    pass

class DiffExp(Exp):
    def __init__(self, f, xs):
        assert(len(xs) > 0)
        self.f = f
        self.xs = xs

    def __str__(self):
        return 'D(' + self.f + ',' + ','.join(self.xs) + ')'

class FunExp(Exp):
    def __init__(self, f, args):
        self.f = f
        self.args = args

    def __str__(self):
        return self.f + '(' + ','.join(map(str,self.args)) + ')'

class GradExp(Exp):
    def __init__(self, x):
        assert(isinstance(x, FieldExp))
        self.arity = x.f.arity

    def __eq__(self, other):
        assert self.arity==len(other)
        return Eq(self, other)

class Eq:
    def __init__(self, x, y):
        self.x = x
        self.y = y

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

def D(f, x, order=1):
    return DiffExp(f, [x for i in range(order)])

class SpatialCoordinate(Exp):
    pass

class TimeCoordinate(Exp):
    pass

def SpatialCoordinates(n):
    return [ SpatialCoordinate() for i in range(n) ]

class FieldHandle:
    def __init__(self, arity):
        self.arity = arity

def Field(coords, m):
    f = FieldHandle(len(coords))
    return [ FieldExp(f,i) for i in range(m) ]
