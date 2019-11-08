import os

DIR = os.path.join(os.path.dirname(__file__), 'builtins')


operations = tuple(l.rstrip().lower() for l in open(os.path.join(DIR, 'operations.txt')).readlines())
functions = tuple(l.rstrip().lower() for l in open(os.path.join(DIR, 'functions.txt')).readlines())
keywords = tuple(l.rstrip().lower() for l in open(os.path.join(DIR, 'keywords.txt')).readlines())
variables = tuple('k%d' % i for i in range(20)) + ('veclen',)
