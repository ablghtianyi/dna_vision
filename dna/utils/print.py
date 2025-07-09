import os
import sys
from functools import wraps

def disable_prints(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Save the current stdout so you can restore it later.
        original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        try:
            return func(*args, **kwargs)
        finally:
            sys.stdout.close()
            sys.stdout = original_stdout
    return wrapper
