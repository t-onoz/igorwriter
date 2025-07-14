import importlib.resources

BUILTIN_DIR = importlib.resources.files("igorwriter").joinpath("builtins")

with BUILTIN_DIR.joinpath("operations.txt").open("r") as f:
    operations = tuple(ln.rstrip().lower() for ln in f.readlines())
with BUILTIN_DIR.joinpath("functions.txt").open("r") as f:
    functions = tuple(ln.rstrip().lower() for ln in f.readlines())
with BUILTIN_DIR.joinpath("keywords.txt").open("r") as f:
    keywords = tuple(ln.rstrip().lower() for ln in f.readlines())
variables = tuple("k%d" % i for i in range(20)) + ("veclen",)
