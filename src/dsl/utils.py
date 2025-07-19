from .matrix import Matrix
from .operations import Operation

def var_names(self, frame_locals):
    for name, var in frame_locals.items():
        if isinstance(var, Matrix):
            if not hasattr(var, 'name') or var.name == "unnamed" or var.name is None:
                var.name = name
