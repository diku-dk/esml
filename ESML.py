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
    def __init__(self, domain, v):
        self.domain = domain
        self.v = v

    def __str__(self):
        return str(self.v)

    def __hash__(self):
        return id(self)

    def leaves(self):
        return [self]

    def eval(self, vtable):
        return vtable[self.v]

    def diff(self, vtable, wrt):
        M = slice(1,-1)
        L = slice(0,-2)
        R = slice(2,None)
        f = vtable[self]
        dx, that, this = finite_differences_slice(self.f, wrt)
        return (1/dx)*(f[that] - f[this])

class DiffExp(Exp):
    def __init__(self, f, xs):
        self.f = f
        self.xs = xs

    def __str__(self):
        return 'D(' + str(self.f) + ',' + ','.join(map(str,self.xs)) + ')'

    def leaves(self):
        return self.f.leaves() + [e for x in self.xs for e in x.leaves()]

    def eval(self, vtable):
        return self.f.diff(vtable, *self.xs)

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

class Flux(Exp):
    def __init__(self, x):
        assert(isinstance(x, FieldExp))
        self.v = x

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
    return coords

def Field(inputs, vs):
    outputs = list([ FieldExp(inputs, v) for v in vs ])
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

class Boundary:
    pass

class MinOf(Boundary):
    def __init__(self, x):
        assert(isinstance(x, SpatialCoordinate))
        self.v = x

class MaxOf(Boundary):
    def __init__(self, x):
        assert(isinstance(x, SpatialCoordinate))
        self.v = x

def fields_in_exp(e):
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

def finite_differences_slice(f, wrt):
    M = slice(1,-1)
    L = slice(0,-2)
    R = slice(2,None)
    assert(len(f.space.coords) == 2)
    dx = 1 # FIXME
    if wrt == f.space.coords[0]:
        return (dx, (R,M), (M,M))
    elif wrt == f.space.coords[1]:
        return (dx, (M,R), (M,M))
    else:
        raise Exception('%a %a', f, wrt)

class ODESystem:
    dtype = np.float64

    def __init__(self, domain, initial, boundary, time, equations, consts={}):
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
        self.space = set([ c for f in self.fields for c in f.domain])
        self.time = time
        self.equations = equations
        self.consts = consts
        self.boundary = boundary

    def describe(self):
        for (c,v) in self.consts.items():
            print('const %s %f.' % (c, v))
        print('time %s.' % self.time.v)
        for c in self.space:
            lo,hi = self.domain.bounds(c)
            print('space (%d <= %s <= %d).' % (lo,c.v,hi))
        for f in self.fields:
            def coord(c):
                return c.v
            def out(o):
                return '(%s = %s)' % (o.v, str(self.initial.initial(o)))
            print('field %s -> %s.' % (' '.join(map(coord,f.domain)), out(f)))
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
            unit = np.ones([dim_steps[c] for c in f.space.coords], dtype=self.dtype)
            for out in f.outputs:
                field_data[out] = unit * self.initial.initial(out).eval(vtable=spatial_grid)

        t = 0
        t0 = 0

        evolve_in_time = set()

        for e in self.equations:
            lhs = e.x
            rhs = e.y

            if (lhs.xs == [self.time]):
                evolve_in_time += lhs.f

        dt = 0.001

class FiniteDifference:
    def __init__(self, type, order):
        self.type = type
        self.order = order

class Integrator:
    def __init__(self, type):
        self.type = type

from io import StringIO

def gen_futhark(system, derivative):
    def field_type(f):
        shape = ''.join([ '[{}]'.
                          format(system.domain.size(d))
                          for d in f.domain ])
        return '{}f64'.format(shape)

    def field_param(f):
        return '{}: {}'.format(f, field_type(f))

    def compile_exp(e):
        if isinstance(e, ConstExp):
            return e.v
        if isinstance(e, FieldExp):
            return e.v
        elif isinstance(e, BinOpExp):
            return '({} {} {})'.format(compile_exp(e.x), e.op, compile_exp(e.y))
        elif isinstance(e, FunExp):
            return '({} {})'.format(e.f, ' '.join(map(compile_exp, e.args)))
        elif isinstance(e, DiffExp):
            return 'D({})'.format(compile_exp(e.f))
        else:
            raise Exception('Cannot compile expression of type {}: {}'.format(type(e),e))

    def field_delta_function(f,rhs):
        rettype = field_type(f)
        idx_params = ['({}: i64)'.format(p) for p in f.domain]
        state_params = ['({})'.format(field_param(f)) for f in system.fields]
        params = state_params + idx_params + ['(dt: f64)']
        body = compile_exp(rhs)
        return 'def d{}_dt {} : {} = {}'.format(f, ' '.join(params), rettype, body)

    io = StringIO()
    for (c,v) in system.consts.items():
        io.write('def {} : f64 = {}\n'.format(c,v))

    for e in system.equations:
        lhs = e.x
        rhs = e.y
        f = lhs.f
        assert(lhs.xs == [system.time])
        io.write(field_delta_function(f,rhs) + '\n')

    return io.getvalue()

class Discretisation:
    def __init__(self, system, spatial_derivative, time_integrator):
        self.system = system
        self.spatial_derivative = spatial_derivative

    def gen(self):
        print(gen_futhark(self.system, self.spatial_derivative))
