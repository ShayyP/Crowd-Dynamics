# General utilities library

def clamp(n, minn, maxn):
    """Simple method that clamps float between two values"""
    return max(min(maxn, n), minn)


def sign(n):
    if n < 0:
        return -n
    return n
