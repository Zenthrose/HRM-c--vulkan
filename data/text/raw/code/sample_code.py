# Python examples for reasoning and problem solving

def binary_search(arr, target):
    """
    Binary search algorithm - demonstrates algorithmic reasoning
    """
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = (left + right) // 2

        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1

def fibonacci_recursive(n):
    """
    Fibonacci sequence - recursive reasoning
    """
    if n <= 1:
        return n
    return fibonacci_recursive(n-1) + fibonacci_recursive(n-2)

def fibonacci_iterative(n):
    """
    Fibonacci sequence - iterative approach
    More efficient for large n due to reasoning about computational complexity
    """
    if n <= 1:
        return n

    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

def is_prime(num):
    """
    Primality testing - mathematical reasoning
    """
    if num <= 1:
        return False
    if num <= 3:
        return True
    if num % 2 == 0 or num % 3 == 0:
        return False

    i = 5
    while i * i <= num:
        if num % i == 0 or num % (i + 2) == 0:
            return False
        i += 6

    return True

def gcd(a, b):
    """
    Greatest Common Divisor - Euclidean algorithm
    Demonstrates mathematical reasoning and recursion
    """
    while b != 0:
        a, b = b, a % b
    return a

def sieve_of_eratosthenes(n):
    """
    Sieve of Eratosthenes - efficient prime generation
    Algorithmic reasoning for optimization
    """
    primes = []
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False

    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n+1, i):
                is_prime[j] = False

    for i in range(2, n+1):
        if is_prime[i]:
            primes.append(i)

    return primes

# Example usage and reasoning
if __name__ == "__main__":
    # Binary search demonstration
    arr = [1, 3, 5, 7, 9, 11, 13, 15]
    target = 7
    result = binary_search(arr, target)
    print(f"Binary search: {target} found at index {result}")

    # Fibonacci comparison
    n = 10
    recursive_result = fibonacci_recursive(n)
    iterative_result = fibonacci_iterative(n)
    print(f"Fibonacci {n}: recursive={recursive_result}, iterative={iterative_result}")

    # Prime checking
    test_numbers = [2, 3, 4, 5, 16, 17, 18, 19, 23]
    for num in test_numbers:
        print(f"{num} is {'prime' if is_prime(num) else 'not prime'}")

    # GCD calculation
    print(f"GCD of 48 and 18 is {gcd(48, 18)}")

    # Prime generation
    primes = sieve_of_eratosthenes(50)
    print(f"Primes up to 50: {primes}")
