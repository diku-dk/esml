import numpy as np

def arg(e):
    if isinstance(e, Exp):
        return e
    else:
        return LiteralExp(e)

class Exp:
    def __add__(self, y):
        return BinOpExp('+', self, arg(y))
    def __radd__(self, x):
        return BinOpExp('+', arg(x), self)
    def __sub__(self, y):
        return BinOpExp('-', self, arg(y))
    def __rsub__(self, x):
        return BinOpExp('-', arg(x), self)
    def __mul__(self, y):
        return BinOpExp('*', self, arg(y))
    def __rmul__(self, x):
        return BinOpExp('*', arg(x), self)
    def __truediv__(self, y):
        return BinOpExp('/', self, arg(y))
    def __rtruediv__(self, x):
        return BinOpExp('/', arg(x), self)
    def __pow__(self, y):
        return BinOpExp('**', self, arg(y))
    def __rpow__(self, x):
        return BinOpExp('**', arg(x), self)
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

    def leaves(self):
        return self.x.leaves() + self.y.leaves()

    def __str__(self):
        return '(' + str(self.x) + str(self.op) + str(self.y) + ')'

    def eval(self, vtable):
        x = self.x.eval(vtable)
        y = self.y.eval(vtable)
        if self.op == '+':
            return x + y
        elif self.op == '-':
            return x - y
        elif self.op == '*':
            return x * y
        elif self.op == '/':
            return x / y
        elif self.op == '**':
            return x ** y
        else:
            return Exception('Cannot eval binop: %s' % self.op)

class LiteralExp(Exp):
    def __init__(self, v):
        self.v = v

    def __str__(self):
        return str(self.v)

    def leaves(self):
        return [self]

    def eval(self, vtable):
        return self.v

class ConstExp(Exp):
    def __init__(self, v):
        self.v = v

    def __str__(self):
        return str(self.v)

    def leaves(self):
        return [self]

    def eval(self, vtable):
        return vtable[self.v]

class FieldExp(Exp):
    def __init__(self, v):
        self.v = v

    def __str__(self):
        return str(self.v)

    def __hash__(self):
        return id(self)

    def leaves(self):
        return [self]

    def eval(self, vtable):
        return vtable[self.v]

class DiffExp(Exp):
    def __init__(self, f, xs):
        self.f = f
        self.xs = xs

    def __str__(self):
        return 'D(' + str(self.f) + ',' + ','.join(map(str,self.xs)) + ')'

    def leaves(self):
        return self.f.leaves() + [e for x in self.xs for e in x.leaves()]

class FunExp(Exp):
    def __init__(self, f, args):
        assert type(args) == list or type(args) == tuple
        self.f = f
        self.args = args

    def __str__(self):
        return str(self.f) + '(' + ','.join(map(str,self.args)) + ')'

    def leaves(self):
        return [e for x in self.args for e in x.leaves()]

    def eval(self, vtable):
        args = [ e.eval(vtable) for e in self.args ]
        if self.f == 'exp':
            return np.exp(*args)
        if self.f == 'neg':
            return -args[0]
        else:
            raise Exception('Cannot eval function: %s' % self.f)

class GradExp(Exp):
    def __init__(self, x):
        assert(isinstance(x, FieldExp))
        self.x = x

    def __eq__(self, other):
        assert len(self.x.f.space.coords)==len(other)
        return Eq(self, arg(other))

    def __str__(self):
        return 'Gradient(' + str(self.x) + ')'

    def leaves(self):
        return [self.x]

class Eq:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        return str(self.x) + ' == ' + str(self.y)

    def __repr__(self):
        return str(self)

    def leaves(self):
        return self.x.leaves() + self.y.leaves()

def MkFunExp(op, arity):
    def f(*es):
        assert len(es) == arity
        return FunExp(op, es)
    return f

sin = MkFunExp('sin', 1)
cos = MkFunExp('cos', 1)
tan = MkFunExp('tan', 1)
exp = MkFunExp('exp', 1)
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

class SpaceHandle:
    def __init__(self, coords):
        self.coords = coords

class SpatialCoordinate(Exp):
    def __init__(self, v):
        self.v = v

    def __str__(self):
        return self.v

    def __hash__(self):
        return id(self)

    def leaves(self):
        return [self]

    def eval(self, vtable):
        return vtable[self]

class TimeCoordinate(Exp):
    def __init__(self, v):
        self.v = v

    def __str__(self):
        return self.v

    def __hash__(self):
        return id(self)

    def leaves(self):
        return [self]

    def eval(self, vtable):
        return vtable[self]

def SpatialCoordinates(*vs):
    coords = list([ SpatialCoordinate(v) for v in vs ])
    h = SpaceHandle(coords)
    for c in coords:
        c.space = h
    return coords

class FieldHandle:
    def __init__(self, space, outputs):
        assert(isinstance(space, SpaceHandle))
        self.space = space
        self.outputs = outputs

def Field(inputs, vs):
    outputs = list([ FieldExp(v) for v in vs ])
    f = FieldHandle(inputs[0].space, outputs)
    for o in outputs:
        o.f = f
    if len(outputs) == 1:
        return outputs[0]
    else:
        return outputs

class RectDomain:
    def __init__(self, bounds):
        self._bounds = bounds

    def has_coord(self, s):
        return s in self._bounds

    def bounds(self, s):
        return self._bounds[s]

    def size(self, s):
        a,b = self.bounds(s)
        return int(b-a)

class Initial:
    def __init__(self, initial):
        for k in initial:
            initial[k] = arg(initial[k])
        self._initial = initial

    def initial(self, c):
        return self._initial[c]

def fields_in_exp(e):
    if isinstance(e, FieldExp):
        return set([e.f])
    else:
        return set()

def field_vars_in_exp(e):
    if isinstance(e, FieldExp):
        return set([e])
    else:
        return set()

def times_in_exp(e):
    if isinstance(e, TimeCoordinate):
        return set([e])
    else:
        return set()

def consts_in_exp(e):
    if isinstance(e, ConstExp):
        return set([e.v])
    else:
        return set()

class ODESystem:
    dtype = np.float64

    def __init__(self, domain, initial, time, equations, consts={}):
        fields = set()
        times = set ()
        for e in equations:
            for l in e.leaves():
                fields |= fields_in_exp(l)
                times |= times_in_exp(l)
                for c in consts_in_exp(l):
                    if not c in consts:
                        raise Exception('Missing const: %a' % c)

        assert(len(times) == 1)
        time = (list(times))[0]

        self.domain = domain
        self.initial = initial
        self.fields = fields
        self.spaces = set([ s.space for s in self.fields])
        self.time = time
        self.equations = equations
        self.consts = consts

        for c in self.consts:
            self.consts[c] = arg(self.consts[c])

    def describe(self):
        for (c,v) in self.consts.items():
            print('const %s %f.' % (c, v.v))
        print('time %s.' % self.time.v)
        for s in self.spaces:
            def coord(c):
                lo,hi = self.domain.bounds(c)
                return '(%d <= %s <= %d)' % (lo,c.v,hi)
            print('space %s.' % ' '.join(map(coord,s.coords)))
        for f in self.fields:
            def coord(c):
                return c.v
            def out(o):
                return '(%s = %s)' % (o.v, str(self.initial.initial(o)))
            print('field %s -> %s.' % (' '.join(map(coord,f.space.coords)),
                                       ' '.join(map(out,f.outputs))))
        for e in self.equations:
            print('equation %s.' % str(e))

    def run(self):
        # Initialise each field.
        field_data = {}

        dim_steps = {}
        dim_linspace = {}
        spatial_grid = {}

        for s in self.spaces:
            for c in s.coords:
                # Hardcode discretisation steps.
                dim_steps[c] = 1000
                lo,hi = self.domain.bounds(c)
                dim_linspace[c] = np.linspace(lo,hi,dim_steps[c], dtype=self.dtype)
            for (c,g) in zip(s.coords, np.meshgrid(*[dim_linspace[c] for c in s.coords])):
                spatial_grid[c] = g

        for f in self.fields:
            for out in f.outputs:
                field_data[out] = np.ones([dim_steps[c] for c in f.space.coords],
                                          dtype=self.dtype) * \
                                          self.initial.initial(out).eval(spatial_grid)
        print(field_data)
