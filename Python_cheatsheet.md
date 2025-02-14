Python Cheatsheet 💻🚀
===============================

**Contents**
--------
**Python Types:** **[`Numbers`](#numbers)__,__[`Strings`](#strings)__,__[`Boolean`](#boolean)__,__[`Lists`](#lists)__,__[`Dictionaries`](#dictionaries)__,__ [`Tuples`](#tuples)__,__[`Sets`](#sets)__,__[`None`](#none)**  

**Python Basics:** **[`Comparison Operators`](#comparison-operators)__,__[`Logical Operators`](#logical-operators)__,__[`Loops`](#loops)__,__[`Range`](#range)__,__[`Enumerate`](#enumerate)__,__[`Counter`](#counter)__,__[`Named Tuple`](#named-tuple)__,__[`OrderedDict`](#ordereddict)**    

**Functions:** **[`Functions`](#functions)__,__[`Lambda`](#lambda)__,__[`Comprehensions`](#comprehensions)__,__[`Map,Filter,Reduce`](#map-filter-reduce)__,__[`Ternary`](#ternary-condition)__,__[`Any,All`](#any-all)__,__[`Closures`](#closures)__,__[`Scope`](#scope)**    

**Advanced Python:** **[`Modules`](#modules)__,__[`Iterators`](#iterators)__,__[`Generators`](#generators)__,__[`Decorators`](#decorators)__,__[`Class`](#class)__,__[`Exceptions`](#exceptions)__,__[`Command Line Arguments`](#command-line-arguments)__,__[`File IO`](#file-io)__,__[`Useful Libraries`](#useful-libraries)**  


Numbers
----
**python's 2 main types for Numbers is int and float (or integers and floating point numbers)**
```python
type(1)   # int 
type(-10) # int
type(0)   # int
type(0.0) # float
type(2.2) # float
type(4E2) # float - 4*10 to the power of 2
```


# Arithmetic

```python
10 + 3  # 13
10 - 3  # 7
10 * 3  # 30
10 ** 3 # 1000
10 / 3  # 3.3333333333333335
10 // 3 # 3 --> floor division - no decimals and returns an int
10 % 3  # 1 --> modulo operator - return the remainder. Good for deciding if number is even or odd
```
# Basic Functions

```python

pow(5, 2)      # 25 --> like doing 5**2
abs(-50)       # 50
round(5.46)    # 5
round(5.468, 2)# 5.47 --> round to nth digit
bin(512)       # '0b1000000000' -->  binary format
hex(512)       # '0x200' --> hexadecimal format
```
# Converting Strings to Numbers

```python

age = input("How old are you?")
age = int(age)
pi = input("What is the value of pi?")
pi = float(pi)
``` 

### Advanced Arithmetic Operations
# Complex Numbers
```python

x = 1 + 2j  # Complex number
type(x)     # <class 'complex'>
x.real      # 1.0
x.imag      # 2.0
```

# Bitwise Operations

```python
a = 10  # 1010 in binary
b = 4   # 0100 in binary
a & b   # 0  --> Bitwise AND
a | b   # 14 --> Bitwise OR
a ^ b   # 14 --> Bitwise XOR
~a      # -11 --> Bitwise NOT
a << 2  # 40 --> Left shift
a >> 2  # 2  --> Right shift
```

### Advanced Arithmetic Operations
```python
# Complex Numbers
x = 1 + 2j  # Complex number
type(x)     # <class 'complex'>
x.real      # 1.0
x.imag      # 2.0

# Bitwise Operations
a = 10  # 1010 in binary
b = 4   # 0100 in binary
a & b   # 0  --> Bitwise AND
a | b   # 14 --> Bitwise OR
a ^ b   # 14 --> Bitwise XOR
~a      # -11 --> Bitwise NOT
a << 2  # 40 --> Left shift
a >> 2  # 2  --> Right shift
```



### Mathematical Functions
```python
import math

# Constants
math.pi       # 3.141592653589793
math.e        # 2.718281828459045

# Logarithmic Functions
math.log(10)        # 2.302585092994046 --> Natural logarithm
math.log10(100)     # 2.0 --> Base-10 logarithm
math.log2(8)        # 3.0 --> Base-2 logarithm

# Trigonometric Functions
math.sin(math.pi/2) # 1.0
math.cos(math.pi)   # -1.0
math.tan(0)         # 0.0

# Angular Conversion
math.degrees(math.pi) # 180.0 --> Radians to degrees
math.radians(180)     # 3.141592653589793 --> Degrees to radians

# Other Functions
math.sqrt(16)        # 4.0 --> Square root
math.factorial(5)    # 120 --> Factorial
math.gcd(12, 16)     # 4 --> Greatest common divisor
math.ceil(4.3)       # 5 --> Ceiling function
math.floor(4.7)      # 4 --> Floor function
```

### Random Numbers
```python
import random

# Random float between 0 and 1
random.random()

# Random float between a and b
random.uniform(1.5, 2.5)

# Random integer between a and b (inclusive)
random.randint(1, 10)

# Random choice from a sequence
random.choice([1, 2, 3, 4, 5])

# Shuffle a list in place
numbers = [1, 2, 3, 4, 5]
random.shuffle(numbers)
```

### Number Formatting
```python
# Formatting Floats
x = 1234.56789
format(x, '0.2f')  # '1234.57' --> 2 decimal places
format(x, '0.1f')  # '1234.6'  --> 1 decimal place

# Formatting Integers
y = 1234
format(y, '05d')   # '01234' --> Zero-padded to 5 digits

# Scientific Notation
z = 123456789
format(z, '0.2e')  # '1.23e+08' --> Scientific notation with 2 decimal places
```

### Working with Fractions
```python
from fractions import Fraction

# Creating fractions
f1 = Fraction(3, 4)  # 3/4
f2 = Fraction(1, 2)  # 1/2

# Arithmetic with fractions
f1 + f2  # Fraction(5, 4)
f1 * f2  # Fraction(3, 8)
f1 / f2  # Fraction(3, 2)

# Converting to float
float(f1)  # 0.75
```

### Working with Decimals
```python
from decimal import Decimal, getcontext

# Setting precision
getcontext().prec = 6

# Creating decimals
d1 = Decimal('0.1')
d2 = Decimal('0.2')

# Arithmetic with decimals
d1 + d2  # Decimal('0.3')
d1 * d2  # Decimal('0.02')
d1 / d2  # Decimal('0.5')
```

### Number Theory
```python
import math

# Check if a number is prime
def is_prime(n):
    if n <= 1:
        return False
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True

is_prime(29)  # True
is_prime(30)  # False

# Generating prime numbers
def generate_primes(limit):
    primes = []
    for num in range(2, limit):
        if is_prime(num):
            primes.append(num)
    return primes

generate_primes(20)  # [2, 3, 5, 7, 11, 13, 17, 19]
```

### Working with Large Numbers
```python
# Python handles large integers gracefully
large_num = 2**1000  # A very large number
len(str(large_num))  # 302 --> Number of digits

# Factorial of a large number
import math
math.factorial(100)  # 93326215443944152681699238856266700490715968264381621468592963895217599993229915608941463976156518286253697920827223758251185210916864000000000000000000000000
```

---

### 1. **Advanced Mathematical Libraries**
Python has powerful libraries for advanced numerical and mathematical computations.

#### **NumPy** (Numerical Python)
```python
import numpy as np

# Arrays and vectorized operations
arr = np.array([1, 2, 3, 4])
arr * 2  # array([2, 4, 6, 8])

# Matrix operations
matrix = np.array([[1, 2], [3, 4]])
np.linalg.det(matrix)  # Determinant: -2.0
np.linalg.inv(matrix)  # Inverse: array([[-2. ,  1. ], [ 1.5, -0.5]])

# Eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(matrix)

# Random numbers
np.random.rand(5)  # Array of 5 random floats between 0 and 1
np.random.normal(0, 1, 10)  # 10 samples from a normal distribution

# Advanced functions
np.fft.fft([1, 2, 3, 4])  # Fast Fourier Transform
np.polyfit([1, 2, 3], [1, 4, 9], 2)  # Polynomial fitting
```

#### **SciPy** (Scientific Python)
```python
from scipy import integrate, optimize, interpolate

# Numerical integration
result, error = integrate.quad(lambda x: np.sin(x), 0, np.pi)  # Result: 2.0

# Optimization
minimum = optimize.minimize(lambda x: (x - 3)**2, x0=0).x  # Find minimum of (x-3)^2

# Interpolation
x = np.array([1, 2, 3, 4])
y = np.array([1, 4, 9, 16])
f = interpolate.interp1d(x, y, kind='cubic')
f(2.5)  # Interpolated value at x=2.5
```

#### **SymPy** (Symbolic Mathematics)
```python
from sympy import symbols, Eq, solve, diff, integrate, sin

# Symbolic variables
x, y = symbols('x y')
expr = x**2 + 2*x + 1

# Solving equations
solve(Eq(expr, 0), x)  # [-1]

# Differentiation
diff(x**3 + 2*x**2 + x, x)  # 3*x**2 + 4*x + 1

# Integration
integrate(x**2, x)  # x**3/3

# Symbolic limits and series
from sympy import limit, series
limit(sin(x)/x, x, 0)  # 1
series(exp(x), x, 0, 5)  # 1 + x + x**2/2 + x**3/6 + x**4/24 + O(x**5)
```

---

### 2. **Performance Optimization**
For computationally intensive tasks, Python offers tools to optimize performance.

#### **Numba** (Just-In-Time Compilation)
```python
from numba import njit

@njit
def fast_sum(arr):
    total = 0
    for num in arr:
        total += num
    return total

arr = np.arange(1_000_000)
fast_sum(arr)  # Much faster than pure Python
```

#### **Cython** (C Extensions for Python)
Cython allows you to write C-like code for performance-critical sections.
```python
# Save as `example.pyx`
def cython_sum(long[:] arr):
    cdef long total = 0
    cdef int i
    for i in range(arr.shape[0]):
        total += arr[i]
    return total

# Compile with Cython and use in Python
```

#### **Multiprocessing**
For parallel processing:
```python
from multiprocessing import Pool

def square(x):
    return x**2

with Pool(4) as p:
    result = p.map(square, range(10))  # [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
```

---

### 3. **Advanced Number Theory**
Python can handle advanced number theory concepts with ease.

#### **Prime Factorization**
```python
def prime_factors(n):
    i = 2
    factors = []
    while i * i <= n:
        while n % i == 0:
            factors.append(i)
            n = n // i
        i += 1
    if n > 1:
        factors.append(n)
    return factors

prime_factors(100)  # [2, 2, 5, 5]
```

#### **Greatest Common Divisor (GCD) and Least Common Multiple (LCM)**
```python
import math
math.gcd(12, 18)  # 6
math.lcm(12, 18)  # 36
```

#### **Modular Arithmetic**
```python
# Modular exponentiation
pow(2, 10, mod=1000)  # 24 (2^10 % 1000)

# Modular inverse
def modinv(a, m):
    return pow(a, -1, m)  # Python 3.8+

modinv(3, 11)  # 4 (3 * 4 ≡ 1 mod 11)
```

#### **Chinese Remainder Theorem**
```python
from sympy.ntheory.modular import crt

crt([3, 5, 7], [2, 3, 2])  # (23, 105) --> x ≡ 23 mod 105
```

---

### 4. **Advanced Numerical Computations**
#### **Arbitrary Precision Arithmetic**
Python’s `decimal` and `mpmath` libraries allow for arbitrary precision.
```python
from decimal import Decimal, getcontext
getcontext().prec = 100  # Set precision to 100 digits
Decimal(1) / Decimal(7)  # 0.1428571428571428571428571428571428571428571428571428571428571428571428571428571428571428571428571429
```

#### **Complex Numbers**
```python
import cmath
cmath.phase(1 + 1j)  # 0.7853981633974483 (phase in radians)
cmath.polar(1 + 1j)  # (1.4142135623730951, 0.7853981633974483) (magnitude, phase)
```

#### **Special Functions**
```python
from scipy.special import gamma, erf, zeta

gamma(5)  # 24.0 (Gamma function)
erf(1)    # 0.8427007929497149 (Error function)
zeta(2)   # 1.6449340668482264 (Riemann zeta function)
```

---

### 5. **Cryptography and Hashing**
Python’s `hashlib` and `cryptography` libraries are useful for cryptographic operations.
```python
import hashlib

# Hashing
hashlib.sha256(b"hello").hexdigest()  # '2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824'

# Random numbers for cryptography
import secrets
secrets.token_hex(16)  # 'a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6'
```

---

### 6. **Machine Learning and Data Science**
#### **Pandas for Numerical Data**
```python
import pandas as pd

df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
df['C'] = df['A'] + df['B']  # Add columns
df.describe()  # Summary statistics
```

#### **TensorFlow/PyTorch for Deep Learning**
```python
import tensorflow as tf

# Create a tensor
tensor = tf.constant([[1, 2], [3, 4]])
tf.reduce_sum(tensor)  # 10
```

---

Strings
----
**strings in python are stored as sequences of letters in memory**
```python
type('Hellloooooo') # str

'I\'m thirsty'
"I'm thirsty"
"\n" # new line
"\t" # adds a tab

'Hey you!'[4] # y
name = 'Andrei Neagoie'
name[4]     # e
name[:]     # Andrei Neagoie
name[1:]    # ndrei Neagoie
name[:1]    # A
name[-1]    # e
name[::1]   # Andrei Neagoie
name[::-1]  # eiogaeN ierdnA
name[0:10:2]# Ade e
# : is called slicing and has the format [ start : end : step ]

'Hi there ' + 'Timmy' # 'Hi there Timmy' --> This is called string concatenation
'*'*10 # **********
```

```python
# Basic Functions
len('turtle') # 6

# Basic Methods
'  I am alone '.strip()               # 'I am alone' --> Strips all whitespace characters from both ends.
'On an island'.strip('d')             # 'On an islan' --> # Strips all passed characters from both ends.
'but life is good!'.split()           # ['but', 'life', 'is', 'good!']
'Help me'.replace('me', 'you')        # 'Help you' --> Replaces first with second param
'Need to make fire'.startswith('Need')# True
'and cook rice'.endswith('rice')      # True
'still there?'.upper()                # STILL THERE?
'HELLO?!'.lower()                     # hello?!
'ok, I am done.'.capitalize()         # 'Ok, I am done.'
'oh hi there'.count('e')              # 2
'bye bye'.index('e')                  # 2
'oh hi there'.find('i')               # 4 --> returns the starting index position of the first occurrence
'oh hi there'.find('a')               # -1
'oh hi there'.index('a')              # Raises ValueError
```

```python
# String Formatting
name1 = 'Andrei'
name2 = 'Sunny'
print(f'Hello there {name1} and {name2}')       # Hello there Andrei and Sunny - Newer way to do things as of python 3.6
print('Hello there {} and {}'.format(name1, name2))# Hello there Andrei and Sunny
print('Hello there %s and %s' %(name1, name2))  # Hello there Andrei and Sunny --> you can also use %d, %f, %r for integers, floats, string representations of objects respectively
```

```python
# Palindrome check
word = 'reviver'
p = bool(word.find(word[::-1]) + 1)
print(p) # True
```

---

### 1. **Advanced String Manipulation**
#### **String Joining and Splitting**
```python
# Joining strings
words = ['Hello', 'world', '!']
sentence = ' '.join(words)  # 'Hello world !'

# Splitting with multiple delimiters
import re
text = "Hello, world! How are you?"
words = re.split(r'[ ,!?]', text)  # ['Hello', '', 'world', '', 'How', 'are', 'you', '']
words = [word for word in words if word]  # Remove empty strings
```

#### **String Translation**
```python
# Translate characters using a mapping table
translation_table = str.maketrans('aeiou', '12345')
text = "This is a test."
translated_text = text.translate(translation_table)  # 'Th3s 3s 1 t2st.'
```

#### **String Alignment**
```python
# Aligning strings
text = "Hello"
text.ljust(10)       # 'Hello     '
text.rjust(10)       # '     Hello'
text.center(10)      # '  Hello   '
text.center(10, '-') # '--Hello---'
```

#### **String Partitioning**
```python
# Partitioning strings
text = "Hello, world!"
before, sep, after = text.partition(',')  # ('Hello', ',', ' world!')
```

---

### 2. **Regular Expressions (Regex)**
Python’s `re` module is powerful for pattern matching and text manipulation.

#### **Basic Regex**
```python
import re

# Search for a pattern
match = re.search(r'\d+', 'The price is 123 dollars.')
if match:
    print(match.group())  # '123'

# Find all matches
matches = re.findall(r'\d+', 'The prices are 123 and 456 dollars.')  # ['123', '456']

# Replace patterns
new_text = re.sub(r'\d+', 'XXX', 'The price is 123 dollars.')  # 'The price is XXX dollars.'
```

#### **Advanced Regex**
```python
# Named groups
match = re.search(r'(?P<name>\w+) is (?P<age>\d+) years old', 'John is 30 years old.')
if match:
    print(match.group('name'))  # 'John'
    print(match.group('age'))   # '30'

# Lookahead and lookbehind
text = "Hello123World"
result = re.findall(r'\d+(?=World)', text)  # ['123'] (lookahead)
result = re.findall(r'(?<=Hello)\d+', text)  # ['123'] (lookbehind)
```

#### **Compiling Regex**
For better performance with repeated use:
```python
pattern = re.compile(r'\d+')
matches = pattern.findall('The prices are 123 and 456 dollars.')  # ['123', '456']
```

---

### 3. **Unicode and Encoding**
Python strings are Unicode by default, but handling encoding/decoding is crucial for working with external data.

#### **Unicode Characters**
```python
# Unicode strings
text = "你好，世界"  # Chinese for "Hello, world"
print(len(text))  # 5 (characters, not bytes)

# Unicode code points
ord('A')  # 65
chr(65)   # 'A'
```

#### **Encoding and Decoding**
```python
# Encoding to bytes
text = "Hello, world!"
encoded = text.encode('utf-8')  # b'Hello, world!'

# Decoding from bytes
decoded = encoded.decode('utf-8')  # 'Hello, world!'

# Handling different encodings
text = "Café"
encoded = text.encode('latin-1')  # b'Caf\xe9'
decoded = encoded.decode('latin-1')  # 'Café'
```

#### **Handling Encoding Errors**
```python
# Ignore errors
text = "Café"
encoded = text.encode('ascii', errors='ignore')  # b'Caf'

# Replace errors
encoded = text.encode('ascii', errors='replace')  # b'Caf?'
```

---

### 4. **String Performance Optimization**
Strings are immutable in Python, so operations like concatenation can be inefficient. Here’s how to optimize:

#### **Using `join` for Concatenation**
```python
# Inefficient
result = ''
for word in ['Hello', 'world', '!']:
    result += word  # Creates a new string each time

# Efficient
result = ''.join(['Hello', 'world', '!'])
```

#### **Using `io.StringIO` for Building Large Strings**
```python
from io import StringIO

buffer = StringIO()
buffer.write("Hello")
buffer.write(" world!")
result = buffer.getvalue()  # 'Hello world!'
```

#### **Using `f-strings` for Formatting**
```python
name = "Alice"
age = 30
text = f"{name} is {age} years old."  # Fast and readable
```

---

### 5. **Advanced String Formatting**
#### **Format Specifiers**
```python
# Floating point precision
pi = 3.141592653589793
print(f"{pi:.2f}")  # '3.14'

# Padding and alignment
text = "Hello"
print(f"{text:>10}")  # '     Hello'
print(f"{text:<10}")  # 'Hello     '
print(f"{text:^10}")  # '  Hello   '

# Formatting numbers
number = 1234567890
print(f"{number:,}")  # '1,234,567,890'
```

#### **Template Strings**
```python
from string import Template

template = Template("Hello, $name!")
text = template.substitute(name="Alice")  # 'Hello, Alice!'
```

---

### 6. **String Interpolation and Templating**
#### **Using `format_map`**
```python
data = {'name': 'Alice', 'age': 30}
text = "Hello, {name}! You are {age} years old.".format_map(data)  # 'Hello, Alice! You are 30 years old.'
```

#### **Using `string.Template`**
```python
from string import Template

template = Template("Hello, $name! You are $age years old.")
text = template.substitute(name="Alice", age=30)  # 'Hello, Alice! You are 30 years old.'
```

---

### 7. **String-Based Algorithms**
#### **Palindrome Check (Advanced)**
```python
def is_palindrome(s):
    s = ''.join(filter(str.isalnum, s)).lower()  # Remove non-alphanumeric characters
    return s == s[::-1]

print(is_palindrome("A man, a plan, a canal: Panama"))  # True
```

#### **Anagram Check**
```python
from collections import Counter

def is_anagram(s1, s2):
    return Counter(s1) == Counter(s2)

print(is_anagram("listen", "silent"))  # True
```

#### **Longest Common Substring**
```python
def longest_common_substring(s1, s2):
    table = [[0] * (len(s2) + 1) for _ in range(len(s1) + 1)]
    longest = 0
    for i in range(1, len(s1) + 1):
        for j in range(1, len(s2) + 1):
            if s1[i - 1] == s2[j - 1]:
                table[i][j] = table[i - 1][j - 1] + 1
                longest = max(longest, table[i][j])
    return longest

print(longest_common_substring("abcdef", "zbcdf"))  # 3 ('bcd')
```

---


Boolean
----
**True or False. Used in a lot of comparison and logical operations in Python**
```python
bool(True)
bool(False)

# all of the below evaluate to False. Everything else will evaluate to True in Python.
print(bool(None))
print(bool(False))
print(bool(0))
print(bool(0.0))
print(bool([]))
print(bool({}))
print(bool(()))
print(bool(''))
print(bool(range(0)))
print(bool(set()))

# See Logical Operators and Comparison Operators section for more on booleans.
```

### **1. Boolean Basics**
- **Boolean Values**: `True` and `False`.
- **Boolean Type**: `bool` (subclass of `int` where `True == 1` and `False == 0`).

```python
print(type(True))   # <class 'bool'>
print(type(False))  # <class 'bool'>
```

---

### **2. Truthy and Falsy Values**
- **Falsy Values**: Values that evaluate to `False` in a Boolean context.
- **Truthy Values**: Everything else evaluates to `True`.

#### **Falsy Values**
```python
print(bool(None))       # False
print(bool(False))      # False
print(bool(0))          # False
print(bool(0.0))        # False
print(bool([]))         # False (empty list)
print(bool({}))         # False (empty dictionary)
print(bool(()))         # False (empty tuple)
print(bool(''))         # False (empty string)
print(bool(range(0)))   # False (empty range)
print(bool(set()))      # False (empty set)
```

#### **Truthy Values**
```python
print(bool(1))          # True
print(bool(-1))         # True
print(bool(0.1))        # True
print(bool([1, 2]))     # True (non-empty list)
print(bool({'a': 1}))   # True (non-empty dictionary)
print(bool((1, 2)))     # True (non-empty tuple)
print(bool('a'))        # True (non-empty string)
print(bool(range(1)))   # True (non-empty range)
print(bool({1, 2}))     # True (non-empty set)
```

---

### **3. Boolean Operations**
Python provides three Boolean operators: `and`, `or`, and `not`.

#### **`and` Operator**
- Returns the first falsy value or the last truthy value.
- Stops evaluating as soon as a falsy value is found.

```python
print(True and False)   # False
print(False and True)   # False
print(1 and 2)          # 2 (both are truthy, returns the last one)
print(0 and 1)          # 0 (first falsy value)
```

#### **`or` Operator**
- Returns the first truthy value or the last falsy value.
- Stops evaluating as soon as a truthy value is found.

```python
print(True or False)    # True
print(False or True)    # True
print(1 or 2)           # 1 (first truthy value)
print(0 or 1)           # 1 (first truthy value)
print(0 or False)       # False (both are falsy, returns the last one)
```

#### **`not` Operator**
- Returns the opposite Boolean value.

```python
print(not True)         # False
print(not False)        # True
print(not 0)            # True (0 is falsy)
print(not 1)            # False (1 is truthy)
```

---

### **4. Comparison Operators**
Comparison operators return Boolean values (`True` or `False`).

| Operator | Description                  | Example          | Result  |
|----------|------------------------------|------------------|---------|
| `==`     | Equal to                     | `5 == 5`         | `True`  |
| `!=`     | Not equal to                 | `5 != 3`         | `True`  |
| `>`      | Greater than                 | `5 > 3`          | `True`  |
| `<`      | Less than                    | `5 < 3`          | `False` |
| `>=`     | Greater than or equal to     | `5 >= 5`         | `True`  |
| `<=`     | Less than or equal to        | `5 <= 3`         | `False` |

#### **Examples**
```python
print(5 == 5)           # True
print(5 != 3)           # True
print(5 > 3)            # True
print(5 < 3)            # False
print(5 >= 5)           # True
print(5 <= 3)           # False
```

---

### **5. Chaining Comparisons**
You can chain multiple comparisons for concise checks.

```python
x = 5
print(1 < x < 10)       # True (equivalent to 1 < x and x < 10)
print(5 == x == 5.0)    # True
```

---

### **6. Membership Operators**
Membership operators (`in` and `not in`) return Boolean values.

| Operator | Description                  | Example          | Result  |
|----------|------------------------------|------------------|---------|
| `in`     | True if value is in sequence | `'a' in 'abc'`   | `True`  |
| `not in` | True if value is not in sequence | `'d' not in 'abc'` | `True`  |

#### **Examples**
```python
print('a' in 'abc')     # True
print('d' not in 'abc') # True
print(1 in [1, 2, 3])   # True
print(4 not in {1, 2, 3}) # True
```

---

### **7. Identity Operators**
Identity operators (`is` and `is not`) check if two objects are the same (same memory location).

| Operator | Description                  | Example          | Result  |
|----------|------------------------------|------------------|---------|
| `is`     | True if both objects are the same | `x is y`       | Depends |
| `is not` | True if both objects are not the same | `x is not y` | Depends |

#### **Examples**
```python
x = [1, 2, 3]
y = [1, 2, 3]
z = x

print(x is y)           # False (different objects)
print(x is z)           # True (same object)
print(x is not y)       # True
```

---

### **8. Short-Circuit Evaluation**
- **`and`**: Stops evaluating as soon as a falsy value is found.
- **`or`**: Stops evaluating as soon as a truthy value is found.

```python
def expensive_operation():
    print("Expensive operation executed!")
    return True

# Short-circuit with `and`
print(False and expensive_operation())  # False (expensive_operation not called)

# Short-circuit with `or`
print(True or expensive_operation())    # True (expensive_operation not called)
```

---

### **9. Boolean Conversion**
Use `bool()` to explicitly convert values to Boolean.

```python
print(bool(10))         # True
print(bool(0))          # False
print(bool('Hello'))    # True
print(bool(''))         # False
```

---

### **10. Practical Use Cases**
- **Conditional Statements**:
  ```python
  if x > 0:
      print("Positive")
  else:
      print("Non-positive")
  ```

- **Filtering**:
  ```python
  numbers = [1, 0, 5, 0, 10]
  filtered = list(filter(bool, numbers))  # [1, 5, 10]
  ```

- **Logical Checks**:
  ```python
  is_valid = True if x > 0 else False
  ```

---

Lists
----
**Unlike strings, lists are mutable sequences in python**
```python
my_list = [1, 2, '3', True]# We assume this list won't mutate for each example below
len(my_list)               # 4
my_list.index('3')         # 2
my_list.count(2)           # 1 --> count how many times 2 appears

my_list[3]                 # True
my_list[1:]                # [2, '3', True]
my_list[:1]                # [1]
my_list[-1]                # True
my_list[::1]               # [1, 2, '3', True]
my_list[::-1]              # [True, '3', 2, 1]
my_list[0:3:2]             # [1, '3']

# : is called slicing and has the format [ start : end : step ]
```

```python
# Add to List
my_list * 2                # [1, 2, '3', True, 1, 2, '3', True]
my_list + [100]            # [1, 2, '3', True, 100] --> doesn't mutate original list, creates new one
my_list.append(100)        # None --> Mutates original list to [1, 2, '3', True, 100]          # Or: <list> += [<el>]
my_list.extend([100, 200]) # None --> Mutates original list to [1, 2, '3', True, 100, 200]
my_list.insert(2, '!!!')   # None -->  [1, 2, '!!!', '3', True] - Inserts item at index and moves the rest to the right.

' '.join(['Hello','There'])# 'Hello There' --> Joins elements using string as separator.
```

```python
# Copy a List
basket = ['apples', 'pears', 'oranges']
new_basket = basket.copy()
new_basket2 = basket[:]
```
```python
# Remove from List
[1,2,3].pop()    # 3 --> mutates original list, default index in the pop method is -1 (the last item)
[1,2,3].pop(1)   # 2 --> mutates original list
[1,2,3].remove(2)# None --> [1,3] Removes first occurrence of item or raises ValueError.
[1,2,3].clear()  # None --> mutates original list and removes all items: []
del [1,2,3][0]   # None --> removes item on index 0 or raises IndexError
```

```python
# Ordering
[1,2,5,3].sort()         # None --> Mutates list to [1, 2, 3, 5]
[1,2,5,3].sort(reverse=True) # None --> Mutates list to [5, 3, 2, 1]
[1,2,5,3].reverse()      # None --> Mutates list to [3, 5, 2, 1]
sorted([1,2,5,3])        # [1, 2, 3, 5] --> new list created
my_list = [(4,1),(2,4),(2,5),(1,6),(8,9)]
sorted(my_list,key=lambda x: int(x[0])) # [(1, 6), (2, 4), (2, 5), (4, 1), (8, 9)] --> sort the list by 1st (0th index) value of the tuple
list(reversed([1,2,5,3]))# [3, 5, 2, 1] --> reversed() returns an iterator
```

```python
# Useful operations
1 in [1,2,5,3]  # True
min([1,2,3,4,5])# 1
max([1,2,3,4,5])# 5
sum([1,2,3,4,5])# 15
```

```python
# Get First and Last element of a list
mList = [63, 21, 30, 14, 35, 26, 77, 18, 49, 10]
first, *x, last = mList
print(first) #63
print(last) #10
```

```python
# Matrix
matrix = [[1,2,3], [4,5,6], [7,8,9]]
matrix[2][0] # 7 --> Grab first first of the third item in the matrix object

# Looping through a matrix by rows:
mx = [[1,2,3],[4,5,6]]
for row in range(len(mx)):
	for col in range(len(mx[0])):
		print(mx[row][col]) # 1 2 3 4 5 6
    
# Transform into a list:
[mx[row][col] for row in range(len(mx)) for col in range(len(mx[0]))] # [1,2,3,4,5,6]

# Combine columns with zip and *:
[x for x in zip(*mx)] # [(1, 3), (2, 4)]

```

```python
# List Comprehensions
# new_list[<action> for <item> in <iterator> if <some condition>]
a = [i for i in 'hello']                  # ['h', 'e', 'l', 'l', '0']
b = [i*2 for i in [1,2,3]]                # [2, 4, 6]
c = [i for i in range(0,10) if i % 2 == 0]# [0, 2, 4, 6, 8]
```

```python
# Advanced Functions
list_of_chars = list('Helloooo')                                   # ['H', 'e', 'l', 'l', 'o', 'o', 'o', 'o']
sum_of_elements = sum([1,2,3,4,5])                                 # 15
element_sum = [sum(pair) for pair in zip([1,2,3],[4,5,6])]         # [5, 7, 9]
sorted_by_second = sorted(['hi','you','man'], key=lambda el: el[1])# ['man', 'hi', 'you']
sorted_by_key = sorted([
                       {'name': 'Bina', 'age': 30},
                       {'name':'Andy', 'age': 18},
                       {'name': 'Zoey', 'age': 55}],
                       key=lambda el: (el['name']))# [{'name': 'Andy', 'age': 18}, {'name': 'Bina', 'age': 30}, {'name': 'Zoey', 'age': 55}]
```

```python
# Read line of a file into a list
with open("myfile.txt") as f:
  lines = [line.strip() for line in f]
```

---

### 1. **Advanced List Creation**
#### **List Comprehensions with Conditions**
```python
# Create a list of squares for even numbers only
squares = [x**2 for x in range(10) if x % 2 == 0]  # [0, 4, 16, 36, 64]
```

#### **Nested List Comprehensions**
```python
# Flatten a 2D list
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
flattened = [num for row in matrix for num in row]  # [1, 2, 3, 4, 5, 6, 7, 8, 9]
```

#### **Using `map` and `filter`**
```python
# Map: Apply a function to all items
numbers = [1, 2, 3, 4]
squared = list(map(lambda x: x**2, numbers))  # [1, 4, 9, 16]

# Filter: Keep items that satisfy a condition
evens = list(filter(lambda x: x % 2 == 0, numbers))  # [2, 4]
```

#### **Using `zip` for Parallel Iteration**
```python
names = ['Alice', 'Bob', 'Charlie']
ages = [25, 30, 35]
combined = list(zip(names, ages))  # [('Alice', 25), ('Bob', 30), ('Charlie', 35)]
```

---

### 2. **Advanced List Manipulation**
#### **List Slicing with Steps**
```python
# Reverse a list
my_list = [1, 2, 3, 4, 5]
reversed_list = my_list[::-1]  # [5, 4, 3, 2, 1]

# Get every second element
every_second = my_list[::2]  # [1, 3, 5]
```

#### **List Unpacking**
```python
# Unpack into variables
first, *middle, last = [1, 2, 3, 4, 5]
print(first)   # 1
print(middle)  # [2, 3, 4]
print(last)    # 5
```

#### **List Concatenation with `itertools.chain`**
```python
from itertools import chain
list1 = [1, 2, 3]
list2 = [4, 5, 6]
combined = list(chain(list1, list2))  # [1, 2, 3, 4, 5, 6]
```

#### **List Rotation**
```python
# Rotate a list to the right by `k` steps
def rotate_list(lst, k):
    k = k % len(lst)
    return lst[-k:] + lst[:-k]

rotate_list([1, 2, 3, 4, 5], 2)  # [4, 5, 1, 2, 3]
```

---

### 3. **Advanced List Operations**
#### **Finding Unique Elements**
```python
# Using `set` (order not preserved)
my_list = [1, 2, 2, 3, 4, 4, 5]
unique = list(set(my_list))  # [1, 2, 3, 4, 5]

# Using `dict.fromkeys` (order preserved)
unique = list(dict.fromkeys(my_list))  # [1, 2, 3, 4, 5]
```

#### **Finding the Most Frequent Element**
```python
from collections import Counter
my_list = [1, 2, 2, 3, 3, 3, 4]
most_common = Counter(my_list).most_common(1)  # [(3, 3)]
```

#### **Grouping Elements**
```python
# Group by even and odd
from itertools import groupby
my_list = [1, 2, 3, 4, 5, 6]
grouped = {k: list(v) for k, v in groupby(sorted(my_list), key=lambda x: x % 2)}
# {0: [2, 4, 6], 1: [1, 3, 5]}
```

#### **Transposing a Matrix**
```python
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
transposed = list(zip(*matrix))  # [(1, 4, 7), (2, 5, 8), (3, 6, 9)]
```

---

### 4. **Performance Optimization**
#### **Using `deque` for Efficient Pop/Append**
```python
from collections import deque
my_list = deque([1, 2, 3, 4])
my_list.appendleft(0)  # [0, 1, 2, 3, 4]
my_list.popleft()      # [1, 2, 3, 4]
```

#### **Avoiding Unnecessary Copies**
```python
# Use slicing or `copy` for shallow copies
original = [1, 2, 3]
copy1 = original[:]
copy2 = original.copy()
```

#### **Using Generators for Large Lists**
```python
# Generator expression
squares = (x**2 for x in range(10))  # Lazy evaluation
for num in squares:
    print(num)
```

---

### 5. **Functional Programming with Lists**
#### **Using `functools.reduce`**
```python
from functools import reduce
numbers = [1, 2, 3, 4]
product = reduce(lambda x, y: x * y, numbers)  # 24
```

#### **Using `itertools` for Advanced Iteration**
```python
from itertools import permutations, combinations, product

# Permutations
perms = list(permutations([1, 2, 3]))  # [(1, 2, 3), (1, 3, 2), (2, 1, 3), ...]

# Combinations
combs = list(combinations([1, 2, 3], 2))  # [(1, 2), (1, 3), (2, 3)]

# Cartesian product
cartesian = list(product([1, 2], ['a', 'b']))  # [(1, 'a'), (1, 'b'), (2, 'a'), (2, 'b')]
```

---

### 6. **List-Based Algorithms**
#### **Finding the Longest Increasing Subsequence (LIS)**
```python
def longest_increasing_subsequence(nums):
    dp = [1] * len(nums)
    for i in range(1, len(nums)):
        for j in range(i):
            if nums[i] > nums[j]:
                dp[i] = max(dp[i], dp[j] + 1)
    return max(dp)

nums = [10, 9, 2, 5, 3, 7, 101, 18]
print(longest_increasing_subsequence(nums))  # 4
```

#### **Finding All Subarrays**
```python
def all_subarrays(arr):
    return [arr[i:j] for i in range(len(arr)) for j in range(i + 1, len(arr) + 1)]

arr = [1, 2, 3]
print(all_subarrays(arr))  # [[1], [1, 2], [1, 2, 3], [2], [2, 3], [3]]
```

---

### 7. **List Utilities**
#### **Chunking a List**
```python
def chunk_list(lst, size):
    return [lst[i:i + size] for i in range(0, len(lst), size)]

lst = [1, 2, 3, 4, 5, 6, 7]
print(chunk_list(lst, 3))  # [[1, 2, 3], [4, 5, 6], [7]]
```

#### **Finding the Index of the Maximum/Minimum Element**
```python
my_list = [10, 20, 30, 40]
max_index = my_list.index(max(my_list))  # 3
min_index = my_list.index(min(my_list))  # 0
```

---


Dictionaries (also known as mappings or hash tables) are built-in data structures that store data in key-value pairs. Starting from Python 3.7, they remember the order of insertion.

## Basic Operations

```python
# Create a dictionary
my_dict = {'name': 'Andrei Neagoie', 'age': 30, 'magic_power': False}

# Accessing values
print(my_dict['name'])        # 'Andrei Neagoie'
print(my_dict.get('age'))     # 30
print(my_dict.get('ages', 0)) # 0 (default if key doesn't exist)

# Dictionary Size
print(len(my_dict))           # 3

# Keys, Values, and Items
print(list(my_dict.keys()))    # ['name', 'age', 'magic_power']
print(list(my_dict.values()))   # ['Andrei Neagoie', 30, False]
print(list(my_dict.items()))     # [('name', 'Andrei Neagoie'), ('age', 30), ('magic_power', False)]
```

## Modifying Dictionaries

```python
# Add new key-value pair
my_dict['favourite_snack'] = 'Grapes'

# Update existing key-value pair
my_dict['age'] = 31

# Remove key-value pairs
del my_dict['name']
my_dict.pop('magic_power', None)  # Remove and return value, if key doesn't exist, return None

# Clear all items
my_dict.clear()  # my_dict will be {}
```

## Merging and Updating Dictionaries

```python
# Update with another dictionary
my_dict.update({'hobby': 'Coding'})  # Add or update multiple items

# Merging dictionaries (Python 3.9+)
new_dict = my_dict | {'language': 'Python'}  # Creates a new merged dictionary

# Merging dictionaries using unpacking
new_dict = {**my_dict, **{'cool': True}}  # Merges and creates a new dictionary
```

## Advanced Dictionary Creation

```python
# Dictionary from a collection of key-value pairs
new_dict = dict([['name', 'Andrei'], ['age', 32], ['magic_power', False]])

# Creating a dictionary from two lists (keys and values)
new_dict = dict(zip(['name', 'age', 'magic_power'], ['Andrei', 32, False]))

# Using dictionary comprehension
squared_dict = {x: x**2 for x in range(5)}  # Create a dict of squares: {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}

# Filtering with dictionary comprehension
filtered_dict = {key: value for key, value in new_dict.items() if key in ['age', 'name']}  # {'name': 'Andrei', 'age': 32}
```

## Iterating Over Dictionaries

```python
# Iterate through keys
for key in my_dict:
    print(key)  # Prints each key

# Iterate through values
for value in my_dict.values():
    print(value)  # Prints each value

# Iterate through key-value pairs
for key, value in my_dict.items():
    print(f"{key}: {value}")  # Prints key-value pairs
```

## Dictionary Views

- **Keys View:** A view object that displays a list of all the keys in the dictionary.
- **Values View:** A view object that displays a list of all the values in the dictionary.
- **Items View:** A view object that displays a list of all the key-value pairs in the dictionary.

```python
keys_view = my_dict.keys()
values_view = my_dict.values()
items_view = my_dict.items()

# These views are dynamic; changes to the dictionary will reflect in them
print(keys_view)    # dict_keys([...])
print(values_view)  # dict_values([...])
print(items_view)   # dict_items([...])
```

## Nested Dictionaries

```python
# Creating a nested dictionary
nested_dict = {
    'person1': {'name': 'Andrei', 'age': 30},
    'person2': {'name': 'Jane', 'age': 25},
}

# Accessing nested dictionary values
print(nested_dict['person1']['name'])  # 'Andrei'

# Modifying nested dictionary values
nested_dict['person2']['age'] = 26
```

## Common Dictionary Methods

| Method               | Description                                |
|----------------------|--------------------------------------------|
| `dict.clear()`       | Remove all items from the dictionary.     |
| `dict.copy()`        | Return a shallow copy of the dictionary.  |
| `dict.fromkeys()`    | Create a new dictionary from keys with specified value. |
| `dict.setdefault()`   | Return the value of the specified key. If it does not exist, insert the key with a specified value. |
| `dict.update()`      | Update the dictionary with elements from another dictionary or from an iterable of key-value pairs. |
| `dict.popitem()`     | Remove and return the last inserted key-value pair. |

## Best Practices

- **Use Descriptive Keys:** Use meaningful key names to make the dictionary self-documenting.
- **Immutable Keys:** Keys must be of immutable types (such as strings, numbers, or tuples).
- **Keep It Flat:** Nested dictionaries can become hard to manage; try to maintain a flatter structure when possible.
- **JSON Compatibility:** Recognize that dictionaries can be converted to and from JSON format using the `json` module (for APIs, configuration files, etc.).

### Example of JSON Conversion

```python
import json

# Dictionary to JSON
json_data = json.dumps(my_dict)

# JSON back to Dictionary
python_dict = json.loads(json_data)
```

-------

### 1. **Advanced Dictionary Creation**
#### **Dictionary Comprehensions**
```python
# Create a dictionary of squares
squares = {x: x**2 for x in range(5)}  # {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}

# Filter dictionary by value
my_dict = {'a': 1, 'b': 2, 'c': 3}
filtered = {k: v for k, v in my_dict.items() if v > 1}  # {'b': 2, 'c': 3}
```

#### **Using `zip` to Create Dictionaries**
```python
keys = ['a', 'b', 'c']
values = [1, 2, 3]
my_dict = dict(zip(keys, values))  # {'a': 1, 'b': 2, 'c': 3}
```

#### **Default Values with `defaultdict`**
```python
from collections import defaultdict

# Default value for missing keys
my_dict = defaultdict(int)
my_dict['a'] += 1  # {'a': 1}
```

#### **Nested Dictionaries**
```python
# Using dictionary comprehensions
nested = {x: {y: x * y for y in range(1, 4)} for x in range(1, 4)}
# {1: {1: 1, 2: 2, 3: 3}, 2: {1: 2, 2: 4, 3: 6}, 3: {1: 3, 2: 6, 3: 9}}
```

---

### 2. **Advanced Dictionary Manipulation**
#### **Merging Dictionaries**
```python
# Using unpacking (Python 3.5+)
dict1 = {'a': 1, 'b': 2}
dict2 = {'b': 3, 'c': 4}
merged = {**dict1, **dict2}  # {'a': 1, 'b': 3, 'c': 4}

# Using `update`
dict1.update(dict2)  # dict1 is now {'a': 1, 'b': 3, 'c': 4}
```

#### **Deep Copying Dictionaries**
```python
import copy
my_dict = {'a': [1, 2, 3]}
deep_copy = copy.deepcopy(my_dict)  # Creates a deep copy
```

#### **Removing Keys Conditionally**
```python
# Remove keys with None values
my_dict = {'a': 1, 'b': None, 'c': 3}
cleaned = {k: v for k, v in my_dict.items() if v is not None}  # {'a': 1, 'c': 3}
```

#### **Inverting a Dictionary**
```python
# Swap keys and values
my_dict = {'a': 1, 'b': 2, 'c': 3}
inverted = {v: k for k, v in my_dict.items()}  # {1: 'a', 2: 'b', 3: 'c'}
```

---

### 3. **Advanced Dictionary Operations**
#### **Finding the Maximum/Minimum Key or Value**
```python
my_dict = {'a': 1, 'b': 2, 'c': 3}

# Max key
max_key = max(my_dict, key=my_dict.get)  # 'c'

# Min value
min_value = min(my_dict.values())  # 1
```

#### **Sorting a Dictionary**
```python
# Sort by key
sorted_by_key = dict(sorted(my_dict.items()))  # {'a': 1, 'b': 2, 'c': 3}

# Sort by value
sorted_by_value = dict(sorted(my_dict.items(), key=lambda item: item[1]))  # {'a': 1, 'b': 2, 'c': 3}
```

#### **Grouping Data with Dictionaries**
```python
from collections import defaultdict

# Group by even and odd
data = [1, 2, 3, 4, 5, 6]
grouped = defaultdict(list)
for num in data:
    key = 'even' if num % 2 == 0 else 'odd'
    grouped[key].append(num)
# {'odd': [1, 3, 5], 'even': [2, 4, 6]}
```

#### **Counting with `Counter`**
```python
from collections import Counter

# Count occurrences
my_list = ['a', 'b', 'a', 'c', 'b', 'a']
count = Counter(my_list)  # {'a': 3, 'b': 2, 'c': 1}
```

---

### 4. **Performance Optimization**
#### **Using `dict.get` for Safe Access**
```python
# Avoid KeyError
value = my_dict.get('d', 'default')  # 'default'
```

#### **Using `dict.setdefault` for Initialization**
```python
# Initialize missing keys
my_dict = {}
my_dict.setdefault('a', []).append(1)  # {'a': [1]}
```

#### **Using `frozenset` for Immutable Keys**
```python
# Use frozenset as dictionary keys
my_dict = {frozenset({'a', 'b'}): 1}
```

---

### 5. **Functional Programming with Dictionaries**
#### **Mapping and Filtering**
```python
# Map values
my_dict = {'a': 1, 'b': 2, 'c': 3}
mapped = {k: v * 2 for k, v in my_dict.items()}  # {'a': 2, 'b': 4, 'c': 6}

# Filter keys
filtered = {k: v for k, v in my_dict.items() if k in {'a', 'b'}}  # {'a': 1, 'b': 2}
```

#### **Using `functools.reduce`**
```python
from functools import reduce

# Sum all values
my_dict = {'a': 1, 'b': 2, 'c': 3}
total = reduce(lambda acc, item: acc + item[1], my_dict.items(), 0)  # 6
```

---

### 6. **Real-World Use Cases**
#### **Counting Word Frequencies**
```python
text = "hello world hello"
words = text.split()
freq = Counter(words)  # {'hello': 2, 'world': 1}
```

#### **Building a Graph**
```python
graph = defaultdict(list)
edges = [('A', 'B'), ('A', 'C'), ('B', 'D')]
for u, v in edges:
    graph[u].append(v)
# {'A': ['B', 'C'], 'B': ['D']}
```

#### **Configurations and Settings**
```python
config = {
    'database': {
        'host': 'localhost',
        'port': 5432,
        'user': 'admin',
        'password': 'secret'
    },
    'logging': {
        'level': 'DEBUG',
        'file': 'app.log'
    }
}
```

---

### 7. **Dictionary-Based Algorithms**
#### **Finding the Longest Substring Without Repeating Characters**
```python
def longest_substring(s):
    char_map = {}
    left = 0
    max_len = 0
    for right, char in enumerate(s):
        if char in char_map and char_map[char] >= left:
            left = char_map[char] + 1
        char_map[char] = right
        max_len = max(max_len, right - left + 1)
    return max_len

print(longest_substring("abcabcbb"))  # 3
```

#### **Two-Sum Problem**
```python
def two_sum(nums, target):
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []

print(two_sum([2, 7, 11, 15], 9))  # [0, 1]
```

---



Tuples
----
**Like lists, but they are used for immutable thing (that don't change)**
```python
my_tuple = ('apple','grapes','mango', 'grapes')
apple, grapes, mango, grapes = my_tuple# Tuple unpacking
len(my_tuple)                          # 4
my_tuple[2]                            # mango
my_tuple[-1]                           # 'grapes'
```

```python
# Immutability
my_tuple[1] = 'donuts'  # TypeError
my_tuple.append('candy')# AttributeError
```

```python
# Methods
my_tuple.index('grapes') # 1
my_tuple.count('grapes') # 2
```

```python
# Zip
list(zip([1,2,3], [4,5,6])) # [(1, 4), (2, 5), (3, 6)]
```

```python
# unzip
z = [(1, 2), (3, 4), (5, 6), (7, 8)] # Some output of zip() function
unzip = lambda z: list(zip(*z))
unzip(z)
```

Sets
---
**Unorderd collection of unique elements.**
```python
my_set = set()
my_set.add(1)  # {1}
my_set.add(100)# {1, 100}
my_set.add(100)# {1, 100} --> no duplicates!
```

```python
new_list = [1,2,3,3,3,4,4,5,6,1]
set(new_list)           # {1, 2, 3, 4, 5, 6}

my_set.remove(100)      # {1} --> Raises KeyError if element not found
my_set.discard(100)     # {1} --> Doesn't raise an error if element not found
my_set.clear()          # {}
new_set = {1,2,3}.copy()# {1,2,3}
```

```python
set1 = {1,2,3}
set2 = {3,4,5}
set3 = set1.union(set2)               # {1,2,3,4,5}
set4 = set1.intersection(set2)        # {3}
set5 = set1.difference(set2)          # {1, 2}
set6 = set1.symmetric_difference(set2)# {1, 2, 4, 5}
set1.issubset(set2)                   # False
set1.issuperset(set2)                 # False
set1.isdisjoint(set2)                 # False --> return True if two sets have a null intersection.

```

```python
# Frozenset
# hashable --> it can be used as a key in a dictionary or as an element in a set.
<frozenset> = frozenset(<collection>)
```

None
----
**None is used for absence of a value and can be used to show nothing has been assigned to an object**
```python
type(None) # NoneType
a = None
```

Comparison Operators
--------
```python
==                   # equal values
!=                   # not equal
>                    # left operand is greater than right operand
<                    # left operand is less than right operand
>=                   # left operand is greater than or equal to right operand
<=                   # left operand is less than or equal to right operand
<element> is <element> # check if two operands refer to same object in memory
```

Logical Operators
--------
```python
1 < 2 and 4 > 1 # True
1 > 3 or 4 > 1  # True
1 is not 4      # True
not True        # False
1 not in [2,3,4]# True

if <condition that evaluates to boolean>:
  # perform action1
elif <condition that evaluates to boolean>:
  # perform action2
else:
  # perform action3
```

Loops
--------
```python
my_list = [1,2,3]
my_tuple = (1,2,3)
my_list2 = [(1,2), (3,4), (5,6)]
my_dict = {'a': 1, 'b': 2. 'c': 3}

for num in my_list:
    print(num) # 1, 2, 3

for num in my_tuple:
    print(num) # 1, 2, 3

for num in my_list2:
    print(num) # (1,2), (3,4), (5,6)

for num in '123':
    print(num) # 1, 2, 3

for idx,value in enumerate(my_list):
    print(idx) # get the index of the item
    print(value) # get the value

for k,v in my_dict.items(): # Dictionary Unpacking
    print(k) # 'a', 'b', 'c'
    print(v) # 1, 2, 3

while <condition that evaluates to boolean>:
  # action
  if <condition that evaluates to boolean>:
    break # break out of while loop
  if <condition that evaluates to boolean>:
    continue # continue to the next line in the block
```

```python
# waiting until user quits
msg = ''
while msg != 'quit':
    msg = input("What should I do?")
    print(msg)
```

Range
-----
```python
range(10)          # range(0, 10) --> 0 to 9
range(1,10)        # range(1, 10)
list(range(0,10,2))# [0, 2, 4, 6, 8]
```

Enumerate
---------
```python
for i, el in enumerate('helloo'):
  print(f'{i}, {el}')
# 0, h
# 1, e
# 2, l
# 3, l
# 4, o
# 5, o
```

Counter
-----
```python
from collections import Counter
colors = ['red', 'blue', 'yellow', 'blue', 'red', 'blue']
counter = Counter(colors)# Counter({'blue': 3, 'red': 2, 'yellow': 1})
counter.most_common()[0] # ('blue', 3)
```

Named Tuple
-----------
* **Tuple is an immutable and hashable list.**
* **Named tuple is its subclass with named elements.**

```python
from collections import namedtuple
Point = namedtuple('Point', 'x y')
p = Point(1, y=2)# Point(x=1, y=2)
p[0]             # 1
p.x              # 1
getattr(p, 'y')  # 2
p._fields        # Or: Point._fields #('x', 'y')
```

```python
from collections import namedtuple
Person = namedtuple('Person', 'name height')
person = Person('Jean-Luc', 187)
f'{person.height}'           # '187'
'{p.height}'.format(p=person)# '187'
```

OrderedDict
--------
* **Maintains order of insertion**
```python
from collections import OrderedDict
# Store each person's languages, keeping # track of who responded first. 
programmers = OrderedDict()
programmers['Tim'] = ['python', 'javascript']
programmers['Sarah'] = ['C++']
programmers['Bia'] = ['Ruby', 'Python', 'Go']

for name, langs in programmers.items():
    print(name + '-->')
    for lang in langs:
      print('\t' + lang)

```

Functions
-------

#### \*args and \*\*kwargs
**Splat (\*) expands a collection into positional arguments, while splatty-splat (\*\*) expands a dictionary into keyword arguments.**
```python
args   = (1, 2)
kwargs = {'x': 3, 'y': 4, 'z': 5}
some_func(*args, **kwargs) # same as some_func(1, 2, x=3, y=4, z=5)
```

#### \* Inside Function Definition
**Splat combines zero or more positional arguments into a tuple, while splatty-splat combines zero or more keyword arguments into a dictionary.**
```python
def add(*a):
    return sum(a)

add(1, 2, 3) # 6
```

##### Ordering of parameters:
```python
def f(*args):                  # f(1, 2, 3)
def f(x, *args):               # f(1, 2, 3)
def f(*args, z):               # f(1, 2, z=3)
def f(x, *args, z):            # f(1, 2, z=3)

def f(**kwargs):               # f(x=1, y=2, z=3)
def f(x, **kwargs):            # f(x=1, y=2, z=3) | f(1, y=2, z=3)

def f(*args, **kwargs):        # f(x=1, y=2, z=3) | f(1, y=2, z=3) | f(1, 2, z=3) | f(1, 2, 3)
def f(x, *args, **kwargs):     # f(x=1, y=2, z=3) | f(1, y=2, z=3) | f(1, 2, z=3) | f(1, 2, 3)
def f(*args, y, **kwargs):     # f(x=1, y=2, z=3) | f(1, y=2, z=3)
def f(x, *args, z, **kwargs):  # f(x=1, y=2, z=3) | f(1, y=2, z=3) | f(1, 2, z=3)
```

#### Other Uses of \*
```python
[*[1,2,3], *[4]]                # [1, 2, 3, 4]
{*[1,2,3], *[4]}                # {1, 2, 3, 4}
(*[1,2,3], *[4])                # (1, 2, 3, 4)
{**{'a': 1, 'b': 2}, **{'c': 3}}# {'a': 1, 'b': 2, 'c': 3}
```

```python
head, *body, tail = [1,2,3,4,5]
```


Lambda
------
```python
# lambda: <return_value>
# lambda <argument1>, <argument2>: <return_value>
```

```python
# Factorial
from functools import reduce

n = 3
factorial = reduce(lambda x, y: x*y, range(1, n+1))
print(factorial) #6
```

```python
# Fibonacci
fib = lambda n : n if n <= 1 else fib(n-1) + fib(n-2)
result = fib(10)
print(result) #55
```

Comprehensions
------
```python
<list> = [i+1 for i in range(10)]         # [1, 2, ..., 10]
<set>  = {i for i in range(10) if i > 5}  # {6, 7, 8, 9}
<iter> = (i+5 for i in range(10))         # (5, 6, ..., 14)
<dict> = {i: i*2 for i in range(10)}      # {0: 0, 1: 2, ..., 9: 18}
```

```python
output = [i+j for i in range(3) for j in range(3)] # [0, 1, 2, 1, 2, 3, 2, 3, 4]

# Is the same as:
output = []
for i in range(3):
  for j in range(3):
    output.append(i+j)
```

Ternary Condition
-------
```python
# <expression_if_true> if <condition> else <expression_if_false>

[a if a else 'zero' for a in [0, 1, 0, 3]] # ['zero', 1, 'zero', 3]
```

Map Filter Reduce
------
```python
from functools import reduce
list(map(lambda x: x + 1, range(10)))            # [1, 2, 3, 4, 5, 6, 7, 8, 9,10]
list(filter(lambda x: x > 5, range(10)))         # (6, 7, 8, 9)
reduce(lambda acc, x: acc + x, range(10))        # 45
```

Any All
------
```python
any([False, True, False])# True if at least one item in collection is truthy, False if empty.
all([True,1,3,True])     # True if all items in collection are true
```


Closures
-------
**We have a closure in Python when:**
* **A nested function references a value of its enclosing function and then**
* **the enclosing function returns the nested function.**

```python
def get_multiplier(a):
    def out(b):
        return a * b
    return out
```

```python
>>> multiply_by_3 = get_multiplier(3)
>>> multiply_by_3(10)
30
```

* **If multiple nested functions within enclosing function reference the same value, that value gets shared.**
* **To dynamically access function's first free variable use `'<function>.__closure__[0].cell_contents'`.**



### Scope
**If variable is being assigned to anywhere in the scope, it is regarded as a local variable, unless it is declared as a 'global' or a 'nonlocal'.**

```python
def get_counter():
    i = 0
    def out():
        nonlocal i
        i += 1
        return i
    return out
```

```python
>>> counter = get_counter()
>>> counter(), counter(), counter()
(1, 2, 3)
```

Modules
----
```python
if __name__ == '__main__': # Runs main() if file wasn't imported.
    main()
```

```python
import <module_name>
from <module_name> import <function_name>
import <module_name> as m
from <module_name> import <function_name> as m_function
from <module_name> import *
```


Iterators
--------
**In this cheatsheet `'<collection>'` can also mean an iterator.**

```python
<iter> = iter(<collection>)
<iter> = iter(<function>, to_exclusive)     # Sequence of return values until 'to_exclusive'.
<el>   = next(<iter> [, default])           # Raises StopIteration or returns 'default' on end.
```


Generators
---------
**Convenient way to implement the iterator protocol.**

```python
def count(start, step):
    while True:
        yield start
        start += step
```

```python
>>> counter = count(10, 2)
>>> next(counter), next(counter), next(counter)
(10, 12, 14)
```


Decorators
---------
**A decorator takes a function, adds some functionality and returns it.**

```python
@decorator_name
def function_that_gets_passed_to_decorator():
    ...
```

**Example Decorator: timing performance using a decorator.**
* **The functools decorator `@functools.wraps` is used to maintain function naming and 
documentation of the function within the decorator.**

```python
from time import time 
import functools

def performance(func):

    @functools.wraps()
    def wrapper(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time() 
        print(f"Took: {t2 - t1} ms")
        return result
    return wrapper

# calling a function with the decorator 
@performance
def long_time():
    print(sum(i*i for i in range(10000)))
``` 

### Debugger Example
**Decorator that prints function's name every time it gets called.**

```python
from functools import wraps

def debug(func):
    @wraps(func)
    def out(*args, **kwargs):
        print(func.__name__)
        return func(*args, **kwargs)
    return out

@debug
def add(x, y):
    return x + y
```
* **Wraps is a helper decorator that copies metadata of function add() to function out().**
* **Without it `'add.__name__'` would return `'out'`.**



Class
-----
**User defined objects are created using the class keyword**

```python
class <name>:
    age = 80 # Class Object Attribute
    def __init__(self, a):
        self.a = a # Object Attribute

    @classmethod
    def get_class_name(cls):
        return cls.__name__
```

### Inheritance
```python
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age  = age

class Employee(Person):
    def __init__(self, name, age, staff_num):
        super().__init__(name, age)
        self.staff_num = staff_num
```

### Multiple Inheritance
```python
class A: pass
class B: pass
class C(A, B): pass
```

**MRO determines the order in which parent classes are traversed when searching for a method:**
```python
>>> C.mro()
[<class 'C'>, <class 'A'>, <class 'B'>, <class 'object'>]
```

Exceptions
----------

```python
try:
  5/0
except ZeroDivisionError:
  print("No division by zero!")
```

```python
while True:
  try:
    x = int(input('Enter your age: '))
  except ValueError:
    print('Oops!  That was no valid number.  Try again...')
  else: # code that depends on the try block running successfully should be placed in the else block.
    print('Carry on!')
    break
```

### Raising Exception
```python
raise ValueError('some error message')
```

### Finally
```python
try:
  raise KeyboardInterrupt
except:
  print('oops')
finally:
  print('All done!')

```



Command Line Arguments
----------------------
```python
import sys
script_name = sys.argv[0]
arguments   = sys.argv[1:]
```

File IO
----
**Opens a file and returns a corresponding file object.**

```python
<file> = open('<path>', mode='r', encoding=None)
```

### Modes
* **`'r'`  - Read (default).**
* **`'w'`  - Write (truncate).**
* **`'x'`  - Write or fail if the file already exists.**
* **`'a'`  - Append.**
* **`'w+'` - Read and write (truncate).**
* **`'r+'` - Read and write from the start.**
* **`'a+'` - Read and write from the end.**
* **`'t'`  - Text mode (default).**
* **`'b'`  - Binary mode.**

### File
```python
<file>.seek(0)                      # Moves to the start of the file.
```

```python
<str/bytes> = <file>.readline()     # Returns a line.
<list>      = <file>.readlines()    # Returns a list of lines.
```

```python
<file>.write(<str/bytes>)           # Writes a string or bytes object.
<file>.writelines(<list>)           # Writes a list of strings or bytes objects.
```
* **Methods do not add or strip trailing newlines.**

### Read Text from File
```python
def read_file(filename):
    with open(filename, encoding='utf-8') as file:
        return file.readlines() # or read()

for line in read_file(filename):
  print(line)
```

### Write Text to File
```python
def write_to_file(filename, text):
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(text)
```

### Append Text to File
```python
def append_to_file(filename, text):
    with open(filename, 'a', encoding='utf-8') as file:
        file.write(text)
```

Useful Libraries
=========

CSV
---
```python
import csv
```

### Read Rows from CSV File
```python
def read_csv_file(filename):
    with open(filename, encoding='utf-8') as file:
        return csv.reader(file, delimiter=';')
```

### Write Rows to CSV File
```python
def write_to_csv_file(filename, rows):
    with open(filename, 'w', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerows(rows)
```


JSON
----
```python
import json
<str>    = json.dumps(<object>, ensure_ascii=True, indent=None)
<object> = json.loads(<str>)
```

### Read Object from JSON File
```python
def read_json_file(filename):
    with open(filename, encoding='utf-8') as file:
        return json.load(file)
```

### Write Object to JSON File
```python
def write_to_json_file(filename, an_object):
    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(an_object, file, ensure_ascii=False, indent=2)
```


Pickle
------
```python
import pickle
<bytes>  = pickle.dumps(<object>)
<object> = pickle.loads(<bytes>)
```

### Read Object from File
```python
def read_pickle_file(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)
```

### Write Object to File
```python
def write_to_pickle_file(filename, an_object):
    with open(filename, 'wb') as file:
        pickle.dump(an_object, file)
```


Profile
-------
### Basic
```python
from time import time
start_time = time()  # Seconds since
...
duration = time() - start_time
```

### Math
```python
from math import e, pi
from math import cos, acos, sin, asin, tan, atan, degrees, radians
from math import log, log10, log2
from math import inf, nan, isinf, isnan
```

### Statistics
```python
from statistics import mean, median, variance, pvariance, pstdev
```

### Random
```python
from random import random, randint, choice, shuffle
random() # random float between 0 and 1
randint(0, 100) # random integer between 0 and 100
random_el = choice([1,2,3,4]) # select a random element from list
shuffle([1,2,3,4]) # shuffles a list
```


Datetime
--------
* **Module 'datetime' provides 'date' `<D>`, 'time' `<T>`, 'datetime' `<DT>` and 'timedelta' `<TD>` classes. All are immutable and hashable.**
* **Time and datetime can be 'aware' `<a>`, meaning they have defined timezone, or 'naive' `<n>`, meaning they don't.**
* **If object is naive it is presumed to be in system's timezone.**

```python
from datetime import date, time, datetime, timedelta
from dateutil.tz import UTC, tzlocal, gettz
```

### Constructors
```python
<D>  = date(year, month, day)
<T>  = time(hour=0, minute=0, second=0, microsecond=0, tzinfo=None, fold=0)
<DT> = datetime(year, month, day, hour=0, minute=0, second=0, ...)
<TD> = timedelta(days=0, seconds=0, microseconds=0, milliseconds=0,
                 minutes=0, hours=0, weeks=0)
```
* **Use `'<D/DT>.weekday()'` to get the day of the week (Mon == 0).**
* **`'fold=1'` means second pass in case of time jumping back for one hour.**

### Now
```python
<D/DTn>  = D/DT.today()                     # Current local date or naive datetime.
<DTn>    = DT.utcnow()                      # Naive datetime from current UTC time.
<DTa>    = DT.now(<tz>)                     # Aware datetime from current tz time.
```

### Timezone
```python
<tz>     = UTC                              # UTC timezone.
<tz>     = tzlocal()                        # Local timezone.
<tz>     = gettz('<Cont.>/<City>')          # Timezone from 'Continent/City_Name' str.
```

```python
<DTa>    = <DT>.astimezone(<tz>)            # Datetime, converted to passed timezone.
<Ta/DTa> = <T/DT>.replace(tzinfo=<tz>)      # Unconverted object with new timezone.
```

Regex
-----
```python
import re
<str>   = re.sub(<regex>, new, text, count=0)  # Substitutes all occurrences.
<list>  = re.findall(<regex>, text)            # Returns all occurrences.
<list>  = re.split(<regex>, text, maxsplit=0)  # Use brackets in regex to keep the matches.
<Match> = re.search(<regex>, text)             # Searches for first occurrence of pattern.
<Match> = re.match(<regex>, text)              # Searches only at the beginning of the text.
```


### Match Object
```python
<str>   = <Match>.group()   # Whole match.
<str>   = <Match>.group(1)  # Part in first bracket.
<tuple> = <Match>.groups()  # All bracketed parts.
<int>   = <Match>.start()   # Start index of a match.
<int>   = <Match>.end()     # Exclusive end index of a match.
```

### Special Sequences
**Expressions below hold true for strings that contain only ASCII characters. Use capital letters for negation.**
```python
'\d' == '[0-9]'          # Digit
'\s' == '[ \t\n\r\f\v]'  # Whitespace
'\w' == '[a-zA-Z0-9_]'   # Alphanumeric
```


Credits
------
Automate the Boring Stuff
with Python by Al Sweigart
