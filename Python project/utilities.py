# General utilities library

def clamp(n, minn, maxn):
    """Simple function that clamps float between two values"""
    return max(min(maxn, n), minn)


def pad(n, length):
    """Simple function that pads a number with 0's to a desired length"""
    while len(n) < length:
        n = '0' + n
    return n


def largest(nums):
    """Simple function that returns the largest number in the array"""
    n = None
    for num in nums:
        if n is None:
            n = num
            continue
        if num > n:
            n = num
    return n
