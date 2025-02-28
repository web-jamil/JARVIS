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
##Dictionary
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
---


---

### **Tuple Methods **

---

### **1. Creating Tuples**
```python
# Empty tuple
my_tuple = ()

# Tuple with elements
my_tuple = (1, 2, 3)

# Single-element tuple (note the trailing comma)
single_tuple = (1,)

# Tuple from a list
my_tuple = tuple([1, 2, 3])  # (1, 2, 3)

# Tuple from a string
my_tuple = tuple("hello")  # ('h', 'e', 'l', 'l', 'o')
```

---

### **2. Accessing Elements**
```python
my_tuple = (10, 20, 30, 40, 50)

# Access by index
print(my_tuple[0])  # 10

# Negative indexing
print(my_tuple[-1])  # 50

# Slicing
print(my_tuple[1:3])  # (20, 30)

# Step slicing
print(my_tuple[::2])  # (10, 30, 50)
```

---

### **3. Tuple Unpacking**
```python
# Basic unpacking
a, b, c = (1, 2, 3)
print(a, b, c)  # 1 2 3

# Unpacking with * for remaining elements
first, *rest = (1, 2, 3, 4, 5)
print(first)  # 1
print(rest)  # [2, 3, 4, 5]

# Unpacking nested tuples
nested_tuple = (1, (2, 3), 4)
a, (b, c), d = nested_tuple
print(a, b, c, d)  # 1 2 3 4
```

---

### **4. Tuple Methods**
Tuples are immutable, so they have only two built-in methods:

#### **`count()`**
```python
# Count occurrences of an element
my_tuple = (1, 2, 2, 3, 4, 2)
count = my_tuple.count(2)  # 3
```

#### **`index()`**
```python
# Find the index of the first occurrence of an element
my_tuple = (10, 20, 30, 20, 40)
index = my_tuple.index(20)  # 1
```

---

### **5. Tuple Operations**
#### **Concatenation**
```python
tuple1 = (1, 2, 3)
tuple2 = (4, 5, 6)
combined = tuple1 + tuple2  # (1, 2, 3, 4, 5, 6)
```

#### **Repetition**
```python
repeated = (1, 2) * 3  # (1, 2, 1, 2, 1, 2)
```

#### **Membership Testing**
```python
my_tuple = (1, 2, 3, 4, 5)
print(3 in my_tuple)  # True
print(6 not in my_tuple)  # True
```

---

### **6. Advanced Tuple Techniques**
#### **Named Tuples**
Named tuples are a subclass of tuples with named fields, making them more readable:
```python
from collections import namedtuple

# Define a named tuple
Point = namedtuple('Point', ['x', 'y'])
p = Point(10, 20)
print(p.x, p.y)  # 10 20
```

#### **Using `zip` with Tuples**
```python
# Combine two lists into a tuple of tuples
keys = ['a', 'b', 'c']
values = [1, 2, 3]
combined = tuple(zip(keys, values))  # (('a', 1), ('b', 2), ('c', 3))
```

#### **Unzipping Tuples**
```python
# Unzip a tuple of tuples
zipped = (('a', 1), ('b', 2), ('c', 3))
keys, values = zip(*zipped)
print(keys)   # ('a', 'b', 'c')
print(values) # (1, 2, 3)
```

---

### **7. Functional Programming with Tuples**
#### **Using `map` and `filter`**
```python
# Map: Apply a function to all items
my_tuple = (1, 2, 3, 4)
squared = tuple(map(lambda x: x**2, my_tuple))  # (1, 4, 9, 16)

# Filter: Keep items that satisfy a condition
evens = tuple(filter(lambda x: x % 2 == 0, my_tuple))  # (2, 4)
```

#### **Using `reduce`**
```python
from functools import reduce

# Sum all elements
my_tuple = (1, 2, 3, 4)
total = reduce(lambda acc, x: acc + x, my_tuple, 0)  # 10
```

---

### **8. Real-World Use Cases**
#### **Returning Multiple Values from Functions**
Tuples are commonly used to return multiple values from a function:
```python
def get_user_info():
    return ("Alice", 30, "New York")

name, age, location = get_user_info()
```

#### **Database Records**
Tuples are often used to represent database records:
```python
user_record = ("Alice", 30, "New York")
```

#### **Dictionary Keys**
Tuples can be used as keys in dictionaries because they are immutable:
```python
my_dict = {('Alice', 30): 'New York', ('Bob', 25): 'San Francisco'}
print(my_dict[('Alice', 30)])  # 'New York'
```

---

### **9. Tuple-Based Algorithms**
#### **Finding the Longest Increasing Subsequence (LIS)**
```python
def longest_increasing_subsequence(nums):
    dp = [1] * len(nums)
    for i in range(1, len(nums)):
        for j in range(i):
            if nums[i] > nums[j]:
                dp[i] = max(dp[i], dp[j] + 1)
    return max(dp)

nums = (10, 9, 2, 5, 3, 7, 101, 18)
print(longest_increasing_subsequence(nums))  # 4
```

#### **Finding All Subarrays**
```python
def all_subarrays(arr):
    return tuple(arr[i:j] for i in range(len(arr)) for j in range(i + 1, len(arr) + 1))

arr = (1, 2, 3)
print(all_subarrays(arr))  # ((1,), (1, 2), (1, 2, 3), (2,), (2, 3), (3,))
```

---

### **10. Advanced Tuple Utilities**
#### **Chunking a Tuple**
```python
def chunk_tuple(t, size):
    return tuple(t[i:i + size] for i in range(0, len(t), size))

my_tuple = (1, 2, 3, 4, 5, 6, 7)
print(chunk_tuple(my_tuple, 3))  # ((1, 2, 3), (4, 5, 6), (7,))
```

#### **Finding the Index of the Maximum/Minimum Element**
```python
my_tuple = (10, 20, 30, 40)
max_index = my_tuple.index(max(my_tuple))  # 3
min_index = my_tuple.index(min(my_tuple))  # 0
```

---

### 1. **Advanced Tuple Creation**
#### **Tuple Comprehensions**
Tuples don’t have comprehensions like lists, but you can use generator expressions and convert them to tuples:
```python
# Create a tuple of squares
squares = tuple(x**2 for x in range(5))  # (0, 1, 4, 9, 16)
```

#### **Named Tuples**
Named tuples are a subclass of tuples with named fields, making them more readable:
```python
from collections import namedtuple

# Define a named tuple
Point = namedtuple('Point', ['x', 'y'])
p = Point(10, 20)
print(p.x, p.y)  # 10 20
```

#### **Using `zip` to Create Tuples**
```python
# Combine two lists into a tuple of tuples
keys = ['a', 'b', 'c']
values = [1, 2, 3]
combined = tuple(zip(keys, values))  # (('a', 1), ('b', 2), ('c', 3))
```

---

### 2. **Advanced Tuple Manipulation**
#### **Tuple Unpacking with `*`**
```python
# Unpack with * for remaining elements
first, *middle, last = (1, 2, 3, 4, 5)
print(first)   # 1
print(middle)  # [2, 3, 4]
print(last)    # 5
```

#### **Swapping Variables**
Tuples make swapping variables easy and concise:
```python
a, b = 10, 20
a, b = b, a  # Swap
print(a, b)  # 20 10
```

#### **Nested Tuples**
Tuples can contain other tuples, making them useful for hierarchical data:
```python
nested = ((1, 2), (3, 4), (5, 6))
print(nested[1][0])  # 3
```

---

### 3. **Advanced Tuple Operations**
#### **Finding the Maximum/Minimum Element**
```python
my_tuple = (10, 20, 30, 40)
max_value = max(my_tuple)  # 40
min_value = min(my_tuple)  # 10
```

#### **Sorting a Tuple**
```python
# Sort a tuple
my_tuple = (3, 1, 4, 1, 5)
sorted_tuple = tuple(sorted(my_tuple))  # (1, 1, 3, 4, 5)
```

#### **Counting and Indexing**
```python
my_tuple = ('a', 'b', 'c', 'a', 'b')
count_a = my_tuple.count('a')  # 2
index_b = my_tuple.index('b')  # 1
```

---

### 4. **Performance Considerations**
#### **Tuples vs Lists**
- **Tuples** are faster than lists for fixed data because they are immutable and have a smaller memory footprint.
- Use tuples for heterogeneous data (e.g., `(name, age, location)`) and lists for homogeneous data (e.g., `[1, 2, 3, 4]`).

#### **Memory Efficiency**
```python
import sys
my_list = [1, 2, 3, 4]
my_tuple = (1, 2, 3, 4)
print(sys.getsizeof(my_list))  # Larger size
print(sys.getsizeof(my_tuple)) # Smaller size
```

---

### 5. **Functional Programming with Tuples**
#### **Using `map` and `filter`**
```python
# Map: Apply a function to all items
my_tuple = (1, 2, 3, 4)
squared = tuple(map(lambda x: x**2, my_tuple))  # (1, 4, 9, 16)

# Filter: Keep items that satisfy a condition
evens = tuple(filter(lambda x: x % 2 == 0, my_tuple))  # (2, 4)
```

#### **Using `reduce`**
```python
from functools import reduce

# Sum all elements
my_tuple = (1, 2, 3, 4)
total = reduce(lambda acc, x: acc + x, my_tuple, 0)  # 10
```

---

### 6. **Real-World Use Cases**
#### **Returning Multiple Values from Functions**
Tuples are commonly used to return multiple values from a function:
```python
def get_user_info():
    return ("Alice", 30, "New York")

name, age, location = get_user_info()
```

#### **Database Records**
Tuples are often used to represent database records:
```python
user_record = ("Alice", 30, "New York")
```

#### **Dictionary Keys**
Tuples can be used as keys in dictionaries because they are immutable:
```python
my_dict = {('Alice', 30): 'New York', ('Bob', 25): 'San Francisco'}
print(my_dict[('Alice', 30)])  # 'New York'
```

---

### 7. **Tuple-Based Algorithms**
#### **Finding the Longest Increasing Subsequence (LIS)**
```python
def longest_increasing_subsequence(nums):
    dp = [1] * len(nums)
    for i in range(1, len(nums)):
        for j in range(i):
            if nums[i] > nums[j]:
                dp[i] = max(dp[i], dp[j] + 1)
    return max(dp)

nums = (10, 9, 2, 5, 3, 7, 101, 18)
print(longest_increasing_subsequence(nums))  # 4
```

#### **Finding All Subarrays**
```python
def all_subarrays(arr):
    return tuple(arr[i:j] for i in range(len(arr)) for j in range(i + 1, len(arr) + 1))

arr = (1, 2, 3)
print(all_subarrays(arr))  # ((1,), (1, 2), (1, 2, 3), (2,), (2, 3), (3,))
```

---

### 8. **Advanced Tuple Utilities**
#### **Chunking a Tuple**
```python
def chunk_tuple(t, size):
    return tuple(t[i:i + size] for i in range(0, len(t), size))

my_tuple = (1, 2, 3, 4, 5, 6, 7)
print(chunk_tuple(my_tuple, 3))  # ((1, 2, 3), (4, 5, 6), (7,))
```

#### **Finding the Index of the Maximum/Minimum Element**
```python
my_tuple = (10, 20, 30, 40)
max_index = my_tuple.index(max(my_tuple))  # 3
min_index = my_tuple.index(min(my_tuple))  # 0
```

---


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

---


---

### **Set Methods Cheatsheet**

---

### **1. Creating Sets**
```python
# Empty set
my_set = set()

# Set with elements
my_set = {1, 2, 3}

# Set from a list
my_set = set([1, 2, 2, 3])  # {1, 2, 3}

# Set from a string
my_set = set("hello")  # {'h', 'e', 'l', 'o'}
```

---

### **2. Adding and Removing Elements**
```python
# Add an element
my_set.add(4)  # {1, 2, 3, 4}

# Remove an element (raises KeyError if not found)
my_set.remove(3)  # {1, 2, 4}

# Discard an element (no error if not found)
my_set.discard(5)  # {1, 2, 4}

# Remove and return an arbitrary element
element = my_set.pop()  # 1 (arbitrary), my_set is now {2, 4}

# Clear all elements
my_set.clear()  # set()
```

---

### **3. Set Operations**
```python
set1 = {1, 2, 3}
set2 = {3, 4, 5}

# Union
union_set = set1 | set2  # {1, 2, 3, 4, 5}
union_set = set1.union(set2)  # Same as above

# Intersection
intersection_set = set1 & set2  # {3}
intersection_set = set1.intersection(set2)  # Same as above

# Difference
difference_set = set1 - set2  # {1, 2}
difference_set = set1.difference(set2)  # Same as above

# Symmetric Difference
symmetric_difference_set = set1 ^ set2  # {1, 2, 4, 5}
symmetric_difference_set = set1.symmetric_difference(set2)  # Same as above
```

---

### **4. Updating Sets**
```python
set1 = {1, 2, 3}
set2 = {3, 4, 5}

# Update with union
set1.update(set2)  # set1 is now {1, 2, 3, 4, 5}

# Update with intersection
set1.intersection_update(set2)  # set1 is now {3}

# Update with difference
set1.difference_update(set2)  # set1 is now {1, 2}

# Update with symmetric difference
set1.symmetric_difference_update(set2)  # set1 is now {1, 2, 4, 5}
```

---

### **5. Set Comparisons**
```python
set1 = {1, 2, 3}
set2 = {1, 2}

# Check if set2 is a subset of set1
is_subset = set2.issubset(set1)  # True

# Check if set1 is a superset of set2
is_superset = set1.issuperset(set2)  # True

# Check if two sets are disjoint (no common elements)
is_disjoint = set1.isdisjoint({4, 5})  # True
```

---

### **6. Copying Sets**
```python
# Shallow copy
new_set = my_set.copy()  # Creates a new set with the same elements
```

---

### **7. Set Comprehensions**
```python
# Create a set of squares
squares = {x**2 for x in range(5)}  # {0, 1, 4, 9, 16}

# Filter elements
evens = {x for x in range(10) if x % 2 == 0}  # {0, 2, 4, 6, 8}
```

---

### **8. Frozensets**
```python
# Create a frozenset (immutable set)
frozen = frozenset([1, 2, 3, 4])

# Frozensets can be used as dictionary keys
my_dict = {frozen: "value"}
```

---

### **9. Membership Testing**
```python
my_set = {1, 2, 3, 4, 5}

# Check if an element is in the set
print(3 in my_set)  # True

# Check if an element is not in the set
print(6 not in my_set)  # True
```

---

### **10. Set Size and Iteration**
```python
my_set = {1, 2, 3, 4, 5}

# Get the number of elements
size = len(my_set)  # 5

# Iterate over elements
for element in my_set:
    print(element)
```

---

### **11. Advanced Set Operations**
#### **Finding Unique Elements**
```python
# Remove duplicates from a list
my_list = [1, 2, 2, 3, 4, 4, 5]
unique_elements = list(set(my_list))  # [1, 2, 3, 4, 5]
```

#### **Filtering Elements**
```python
# Filter even numbers
my_set = {1, 2, 3, 4, 5}
evens = {x for x in my_set if x % 2 == 0}  # {2, 4}
```

#### **Using `map` and `filter`**
```python
# Map: Apply a function to all items
my_set = {1, 2, 3, 4}
squared = set(map(lambda x: x**2, my_set))  # {1, 4, 9, 16}

# Filter: Keep items that satisfy a condition
evens = set(filter(lambda x: x % 2 == 0, my_set))  # {2, 4}
```

---

### **12. Real-World Use Cases**
#### **Finding Unique Words in a Text**
```python
text = "hello world hello"
words = text.split()
unique_words = set(words)  # {'hello', 'world'}
```

#### **Graph Algorithms**
```python
graph = {
    'A': {'B', 'C'},
    'B': {'A', 'D'},
    'C': {'A'},
    'D': {'B'}
}

visited = set()

def dfs(node):
    if node not in visited:
        visited.add(node)
        for neighbor in graph[node]:
            dfs(neighbor)

dfs('A')
print(visited)  # {'A', 'B', 'C', 'D'}
```

---

### **13. Set-Based Algorithms**
#### **Finding the Longest Consecutive Sequence**
```python
def longest_consecutive_sequence(nums):
    num_set = set(nums)
    longest = 0
    for num in num_set:
        if num - 1 not in num_set:
            current_num = num
            current_length = 1
            while current_num + 1 in num_set:
                current_num += 1
                current_length += 1
            longest = max(longest, current_length)
    return longest

nums = [100, 4, 200, 1, 3, 2]
print(longest_consecutive_sequence(nums))  # 4
```

---


### 1. **Advanced Set Creation**
#### **Set Comprehensions**
```python
# Create a set of squares
squares = {x**2 for x in range(5)}  # {0, 1, 4, 9, 16}
```

#### **Using `frozenset` for Immutable Sets**
```python
# Create a frozenset
frozen = frozenset([1, 2, 3, 4])
print(frozen)  # frozenset({1, 2, 3, 4})
```

#### **Creating Sets from Other Collections**
```python
# From a list
my_list = [1, 2, 2, 3, 4]
my_set = set(my_list)  # {1, 2, 3, 4}

# From a string
my_string = "hello"
char_set = set(my_string)  # {'h', 'e', 'l', 'o'}
```

---

### 2. **Advanced Set Manipulation**
#### **Set Operations**
```python
set1 = {1, 2, 3}
set2 = {3, 4, 5}

# Union
union_set = set1 | set2  # {1, 2, 3, 4, 5}

# Intersection
intersection_set = set1 & set2  # {3}

# Difference
difference_set = set1 - set2  # {1, 2}

# Symmetric Difference
symmetric_difference_set = set1 ^ set2  # {1, 2, 4, 5}
```

#### **Updating Sets**
```python
set1 = {1, 2, 3}
set2 = {3, 4, 5}

# Update with union
set1.update(set2)  # set1 is now {1, 2, 3, 4, 5}

# Update with intersection
set1.intersection_update(set2)  # set1 is now {3}

# Update with difference
set1.difference_update(set2)  # set1 is now {1, 2}

# Update with symmetric difference
set1.symmetric_difference_update(set2)  # set1 is now {1, 2, 4, 5}
```

#### **Subset and Superset Checks**
```python
set1 = {1, 2, 3}
set2 = {1, 2}

# Subset
is_subset = set2.issubset(set1)  # True

# Superset
is_superset = set1.issuperset(set2)  # True

# Disjoint
is_disjoint = set1.isdisjoint({4, 5})  # True
```

---

### 3. **Advanced Set Operations**
#### **Finding Unique Elements**
```python
# Remove duplicates from a list
my_list = [1, 2, 2, 3, 4, 4, 5]
unique_elements = list(set(my_list))  # [1, 2, 3, 4, 5]
```

#### **Filtering Elements**
```python
# Filter even numbers
my_set = {1, 2, 3, 4, 5}
evens = {x for x in my_set if x % 2 == 0}  # {2, 4}
```

#### **Using `map` and `filter`**
```python
# Map: Apply a function to all items
my_set = {1, 2, 3, 4}
squared = set(map(lambda x: x**2, my_set))  # {1, 4, 9, 16}

# Filter: Keep items that satisfy a condition
evens = set(filter(lambda x: x % 2 == 0, my_set))  # {2, 4}
```

---

### 4. **Performance Considerations**
#### **Membership Testing**
Sets are optimized for membership testing:
```python
my_set = {1, 2, 3, 4, 5}
print(3 in my_set)  # True
```

#### **Memory Efficiency**
Sets use more memory than lists but provide faster lookups:
```python
import sys
my_list = [1, 2, 3, 4, 5]
my_set = {1, 2, 3, 4, 5}
print(sys.getsizeof(my_list))  # Smaller size
print(sys.getsizeof(my_set))   # Larger size
```

---

### 5. **Functional Programming with Sets**
#### **Using `reduce`**
```python
from functools import reduce

# Union of multiple sets
sets = [{1, 2}, {2, 3}, {3, 4}]
union = reduce(lambda acc, s: acc | s, sets)  # {1, 2, 3, 4}
```

#### **Using `itertools` for Advanced Iteration**
```python
from itertools import combinations

# All possible pairs
my_set = {1, 2, 3}
pairs = set(combinations(my_set, 2))  # {(1, 2), (1, 3), (2, 3)}
```

---

### 6. **Real-World Use Cases**
#### **Finding Unique Words in a Text**
```python
text = "hello world hello"
words = text.split()
unique_words = set(words)  # {'hello', 'world'}
```

#### **Graph Algorithms**
Sets are useful for graph algorithms like finding connected components:
```python
graph = {
    'A': {'B', 'C'},
    'B': {'A', 'D'},
    'C': {'A'},
    'D': {'B'}
}

visited = set()

def dfs(node):
    if node not in visited:
        visited.add(node)
        for neighbor in graph[node]:
            dfs(neighbor)

dfs('A')
print(visited)  # {'A', 'B', 'C', 'D'}
```

#### **Database Query Optimization**
Sets can be used to optimize database queries by reducing the number of unique elements:
```python
query_results = [1, 2, 2, 3, 4, 4, 5]
unique_results = set(query_results)  # {1, 2, 3, 4, 5}
```

---

### 7. **Set-Based Algorithms**
#### **Finding the Longest Consecutive Sequence**
```python
def longest_consecutive_sequence(nums):
    num_set = set(nums)
    longest = 0
    for num in num_set:
        if num - 1 not in num_set:
            current_num = num
            current_length = 1
            while current_num + 1 in num_set:
                current_num += 1
                current_length += 1
            longest = max(longest, current_length)
    return longest

nums = [100, 4, 200, 1, 3, 2]
print(longest_consecutive_sequence(nums))  # 4
```

#### **Finding Common Elements in Multiple Sets**
```python
sets = [{1, 2, 3}, {2, 3, 4}, {3, 4, 5}]
common_elements = set.intersection(*sets)  # {3}
```

---

### 8. **Advanced Set Utilities**
#### **Chunking a Set**
```python
def chunk_set(s, size):
    return [s[i:i + size] for i in range(0, len(s), size)]

my_set = {1, 2, 3, 4, 5, 6, 7}
print(chunk_set(list(my_set), 3))  # [[1, 2, 3], [4, 5, 6], [7]]
```

#### **Finding the Maximum/Minimum Element**
```python
my_set = {10, 20, 30, 40}
max_value = max(my_set)  # 40
min_value = min(my_set)  # 10
```

---

None
----
**None is used for absence of a value and can be used to show nothing has been assigned to an object**
```python
type(None) # NoneType
a = None
```

---

### **1. Basics of `None`**
```python
# None is a singleton object of the NoneType class
type(None)  # <class 'NoneType'>

# Assigning None to a variable
a = None
print(a)  # None
```

---

### **2. Common Use Cases**
#### **Default Value for Functions**
```python
def greet(name=None):
    if name is None:
        return "Hello, Guest!"
    return f"Hello, {name}!"

print(greet())        # Hello, Guest!
print(greet("Alice")) # Hello, Alice!
```

#### **Placeholder for Uninitialized Variables**
```python
result = None
if some_condition:
    result = "Success"
print(result)  # None or "Success" depending on the condition
```

#### **Return Value for Functions That Don’t Explicitly Return**
```python
def do_nothing():
    pass

print(do_nothing())  # None
```

---

### **3. Checking for `None`**
#### **Using `is` for Identity Check**
```python
a = None
if a is None:
    print("a is None")  # a is None
```

#### **Avoid Using `==` for `None` Checks**
```python
a = None
if a == None:  # Works but not recommended
    print("a is None")

# Always use `is` for None checks
if a is None:
    print("a is None")
```

---

### **4. Best Practices**
#### **Use `None` for Optional Parameters**
```python
def process_data(data, callback=None):
    # Process data
    if callback is not None:
        callback(data)
```

#### **Avoid Using `None` as a Sentinel Value**
```python
# Bad practice
def find_index(lst, target):
    for i, val in enumerate(lst):
        if val == target:
            return i
    return None  # Unclear if None means "not found" or "error"

# Better practice
def find_index(lst, target):
    for i, val in enumerate(lst):
        if val == target:
            return i
    raise ValueError("Target not found in list")
```

---

### **5. Advanced Techniques**
#### **Using `None` in Data Structures**
```python
# Initialize a list with None
my_list = [None] * 5  # [None, None, None, None, None]

# Use None as a placeholder in dictionaries
my_dict = {'name': 'Alice', 'age': None}
```

#### **`None` in Conditional Expressions**
```python
# Use None in ternary operations
result = "Valid" if some_condition else None
```

#### **`None` in Type Annotations**
```python
from typing import Optional

def get_user(id: int) -> Optional[str]:
    if id == 1:
        return "Alice"
    return None
```

---

### **6. Common Pitfalls**
#### **Mutable Default Arguments**
```python
# Bad practice
def append_to_list(value, my_list=[]):
    my_list.append(value)
    return my_list

print(append_to_list(1))  # [1]
print(append_to_list(2))  # [1, 2]  # Unexpected behavior!

# Good practice
def append_to_list(value, my_list=None):
    if my_list is None:
        my_list = []
    my_list.append(value)
    return my_list

print(append_to_list(1))  # [1]
print(append_to_list(2))  # [2]
```

#### **`None` in Comparisons**
```python
# Avoid using None in comparisons with other types
a = None
if a == 0:  # False, but unclear intent
    print("a is 0")
```

---

### **7. Real-World Use Cases**
#### **Database Query Results**
```python
# Simulate a database query
def query_database():
    # Assume no results found
    return None

result = query_database()
if result is None:
    print("No results found")
```

#### **Optional Configuration**
```python
# Configuration with optional settings
config = {
    'host': 'localhost',
    'port': 8080,
    'timeout': None  # No timeout
}
```

---

### **8. `None` in Object-Oriented Programming**
#### **Uninitialized Attributes**
```python
class User:
    def __init__(self, name):
        self.name = name
        self.age = None  # Age not provided initially

user = User("Alice")
print(user.age)  # None
```

#### **Optional Method Return**
```python
class Calculator:
    def divide(self, a, b):
        if b == 0:
            return None  # Indicate invalid operation
        return a / b

calc = Calculator()
result = calc.divide(10, 0)
if result is None:
    print("Division by zero error")
```

---

### **9. `None` in Functional Programming**
#### **Using `None` with `filter`**
```python
# Filter out None values
data = [1, None, 2, None, 3]
filtered = list(filter(lambda x: x is not None, data))  # [1, 2, 3]
```

#### **Using `None` with `map`**
```python
# Map with a function that may return None
def safe_divide(a, b):
    return a / b if b != 0 else None

results = list(map(safe_divide, [10, 20, 30], [2, 0, 5]))  # [5.0, None, 6.0]
```

---

### **10. Summary of Key Points**
- **`None` is a singleton object** representing the absence of a value.
- Use **`is None`** for identity checks (not `==`).
- **Avoid using `None` as a sentinel value**; prefer exceptions or specific return values.
- Use `None` for **optional parameters**, **uninitialized variables**, and **placeholder values**.
- Be cautious with **mutable default arguments** in functions.

---
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

---

---

### **1. Basic Comparison Operators**
```python
# Equality
5 == 5  # True
5 == 10 # False

# Inequality
5 != 10 # True
5 != 5  # False

# Greater Than
10 > 5  # True
5 > 10  # False

# Less Than
5 < 10  # True
10 < 5  # False

# Greater Than or Equal To
10 >= 10 # True
10 >= 5  # True
5 >= 10  # False

# Less Than or Equal To
5 <= 5  # True
5 <= 10 # True
10 <= 5 # False
```

---

### **2. Identity Comparison (`is` vs `==`)**
#### **`is` Operator**
- Checks if two operands refer to the **same object in memory**.
- Used for **identity comparison**.

```python
a = [1, 2, 3]
b = a
c = [1, 2, 3]

print(a is b)  # True (same object)
print(a is c)  # False (different objects)
```

#### **`==` Operator**
- Checks if two operands have the **same value**.
- Used for **value comparison**.

```python
a = [1, 2, 3]
b = [1, 2, 3]

print(a == b)  # True (same values)
```

#### **When to Use `is` vs `==`**
- Use `is` for **singleton objects** like `None`, `True`, or `False`.
- Use `==` for **value-based comparisons**.

```python
x = None
if x is None:  # Correct
    print("x is None")

if x == None:  # Avoid (works but not idiomatic)
    print("x is None")
```

---

### **3. Chained Comparisons**
Python allows **chaining comparison operators** for concise and readable code.

```python
# Check if a value is within a range
x = 10
if 5 < x < 15:
    print("x is between 5 and 15")  # True

# Multiple comparisons
a, b, c = 5, 10, 15
if a < b < c:
    print("a < b < c")  # True
```

---

### **4. Advanced Comparisons**
#### **Comparing Sequences**
- Sequences (lists, tuples, strings) are compared **lexicographically**.

```python
# Lists
print([1, 2, 3] < [1, 2, 4])  # True
print([1, 2, 3] > [1, 2, 2])  # True

# Strings
print("apple" < "banana")  # True
print("apple" > "Banana")  # True (lexicographical order, 'A' < 'a')
```

#### **Comparing Custom Objects**
- Use the `__eq__`, `__lt__`, `__gt__`, etc., methods to define custom comparison behavior.

```python
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def __eq__(self, other):
        return self.age == other.age

    def __lt__(self, other):
        return self.age < other.age

alice = Person("Alice", 30)
bob = Person("Bob", 25)

print(alice == bob)  # False
print(alice < bob)   # False
print(alice > bob)   # True
```

---

### **5. Performance Considerations**
#### **Short-Circuiting**
- Comparison operators like `and` and `or` **short-circuit**, meaning they stop evaluating as soon as the result is determined.

```python
# Short-circuiting with `and`
def expensive_operation():
    print("Expensive operation executed")
    return True

if False and expensive_operation():
    print("This won't run")  # expensive_operation() is never called

# Short-circuiting with `or`
if True or expensive_operation():
    print("This will run")  # expensive_operation() is never called
```

#### **Avoid Redundant Comparisons**
```python
# Bad practice
if x > 5 and x < 10:
    pass

# Good practice (use chained comparisons)
if 5 < x < 10:
    pass
```

---

### **6. Real-World Use Cases**
#### **Range Checking**
```python
# Check if a value is within a valid range
temperature = 25
if 0 <= temperature <= 100:
    print("Temperature is within safe limits")
```

#### **Sorting with Custom Comparisons**
```python
# Sort a list of tuples by the second element
data = [(1, 20), (2, 10), (3, 30)]
sorted_data = sorted(data, key=lambda x: x[1])  # [(2, 10), (1, 20), (3, 30)]
```

#### **Filtering Data**
```python
# Filter a list based on a condition
numbers = [10, 20, 30, 40, 50]
filtered = [x for x in numbers if x > 25]  # [30, 40, 50]
```

---

### **7. Advanced Techniques**
#### **Using `operator` Module**
The `operator` module provides function equivalents for comparison operators.

```python
import operator

# Compare two values
print(operator.eq(5, 5))  # True
print(operator.lt(5, 10)) # True

# Sort with operator module
data = [(1, 20), (2, 10), (3, 30)]
sorted_data = sorted(data, key=operator.itemgetter(1))  # [(2, 10), (1, 20), (3, 30)]
```

#### **Partial Comparisons with `functools.partial`**
```python
from functools import partial

# Create a function to check if a number is greater than 10
is_greater_than_10 = partial(operator.lt, 10)
print(is_greater_than_10(15))  # True
print(is_greater_than_10(5))   # False
```

---

### **8. Common Pitfalls**
#### **Floating-Point Comparisons**
- Floating-point numbers can have precision issues. Use a tolerance for comparisons.

```python
# Bad practice
print(0.1 + 0.2 == 0.3)  # False

# Good practice
tolerance = 1e-9
print(abs((0.1 + 0.2) - 0.3) < tolerance)  # True
```

#### **Comparing Different Types**
- Comparing different types can lead to unexpected results or errors.

```python
# Avoid
print("5" == 5)  # False

# Use explicit type conversion
print(int("5") == 5)  # True
```

---

### **9. Summary of Key Points**
- Use `==` for **value comparison** and `is` for **identity comparison**.
- Use **chained comparisons** for concise range checks.
- Define custom comparison behavior using `__eq__`, `__lt__`, etc.
- Leverage **short-circuiting** for performance optimization.
- Be cautious with **floating-point comparisons** and **type mismatches**.

---

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


---

### **1. Basic Logical Operators**
```python
# AND: True if both operands are True
print(1 < 2 and 4 > 1)  # True
print(1 < 2 and 4 < 1)  # False

# OR: True if at least one operand is True
print(1 > 3 or 4 > 1)   # True
print(1 > 3 or 4 < 1)   # False

# NOT: Inverts the boolean value
print(not True)         # False
print(not False)        # True

# IS NOT: Checks if two operands are not the same object
print(1 is not 4)       # True

# NOT IN: Checks if an element is not in a collection
print(1 not in [2, 3, 4])  # True
```

---

### **2. Short-Circuiting Behavior**
Logical operators `and` and `or` **short-circuit**, meaning they stop evaluating as soon as the result is determined.

#### **`and` Short-Circuiting**
- If the first operand is `False`, the second operand is **not evaluated**.

```python
def expensive_operation():
    print("Expensive operation executed")
    return True

if False and expensive_operation():
    print("This won't run")  # expensive_operation() is never called
```

#### **`or` Short-Circuiting**
- If the first operand is `True`, the second operand is **not evaluated**.

```python
if True or expensive_operation():
    print("This will run")  # expensive_operation() is never called
```

---

### **3. Advanced Logical Expressions**
#### **Chaining Logical Operators**
You can chain multiple logical operators for complex conditions.

```python
# Check if a number is between 5 and 10 and even
x = 6
if 5 < x < 10 and x % 2 == 0:
    print("x is between 5 and 10 and even")  # True
```

#### **Using `all()` and `any()`**
- `all()`: Returns `True` if **all elements** in an iterable are `True`.
- `any()`: Returns `True` if **at least one element** in an iterable is `True`.

```python
# Check if all numbers in a list are positive
numbers = [1, 2, 3, 4, 5]
print(all(x > 0 for x in numbers))  # True

# Check if any number in a list is negative
numbers = [1, 2, -3, 4, 5]
print(any(x < 0 for x in numbers))  # True
```

---

### **4. Truthiness and Falsiness**
In Python, values can be evaluated as `True` or `False` in a boolean context.

#### **Falsy Values**
- `False`
- `None`
- `0` (integer or float)
- Empty sequences: `""`, `[]`, `()`, `{}`
- Empty collections: `set()`, `dict()`

#### **Truthy Values**
- Everything else is considered `True`.

```python
# Check if a list is not empty
my_list = []
if my_list:
    print("List is not empty")
else:
    print("List is empty")  # This will run
```

---

### **5. Real-World Use Cases**
#### **Input Validation**
```python
# Validate user input
user_input = input("Enter a number: ")
if user_input.isdigit() and int(user_input) > 0:
    print("Valid input")
else:
    print("Invalid input")
```

#### **Conditional Execution**
```python
# Execute code based on multiple conditions
is_weekend = True
has_time_off = False

if is_weekend or has_time_off:
    print("You can relax!")
else:
    print("Time to work!")
```

#### **Filtering Data**
```python
# Filter a list based on multiple conditions
numbers = [1, 2, 3, 4, 5, 6]
filtered = [x for x in numbers if x > 3 and x % 2 == 0]  # [4, 6]
```

---

### **6. Advanced Techniques**
#### **Using `operator` Module**
The `operator` module provides function equivalents for logical operators.

```python
import operator

# Logical AND
print(operator.and_(True, False))  # False

# Logical OR
print(operator.or_(True, False))   # True

# Logical NOT
print(operator.not_(True))         # False
```

#### **Using `functools.partial` for Logical Conditions**
```python
from functools import partial

# Create a function to check if a number is greater than 10
is_greater_than_10 = partial(operator.gt, 10)
print(is_greater_than_10(15))  # True
print(is_greater_than_10(5))   # False
```

---

### **7. Performance Considerations**
#### **Order of Conditions**
- Place **cheaper conditions** first in `and`/`or` expressions to take advantage of short-circuiting.

```python
# Bad practice
if expensive_operation() and x > 5:
    pass

# Good practice
if x > 5 and expensive_operation():
    pass
```

#### **Avoid Redundant Checks**
```python
# Bad practice
if x > 5 and x < 10 and x != 7:
    pass

# Good practice (combine conditions logically)
if 5 < x < 10 and x != 7:
    pass
```

---

### **8. Common Pitfalls**
#### **Confusing `and`/`or` with `&`/`|`**
- `and`/`or` are **logical operators**.
- `&`/`|` are **bitwise operators**.

```python
# Logical AND
print(True and False)  # False

# Bitwise AND
print(True & False)    # 0
```

#### **Implicit Truthiness Checks**
- Avoid relying on implicit truthiness checks for non-boolean values unless intentional.

```python
# Bad practice
if my_list:  # Unclear if checking for non-empty list or truthiness
    pass

# Good practice
if len(my_list) > 0:  # Explicit check
    pass
```

---

### **9. Summary of Key Points**
- Use `and`, `or`, and `not` for **logical operations**.
- Leverage **short-circuiting** for performance optimization.
- Use `all()` and `any()` for **iterable-based conditions**.
- Be mindful of **truthiness** and **falsiness** in boolean contexts.
- Avoid confusing **logical operators** with **bitwise operators**.

---



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

---

### **1. Basic Loops**
#### **`for` Loop**
```python
# Iterate over a list
my_list = [1, 2, 3]
for num in my_list:
    print(num)  # 1, 2, 3

# Iterate over a tuple
my_tuple = (1, 2, 3)
for num in my_tuple:
    print(num)  # 1, 2, 3

# Iterate over a string
for char in "123":
    print(char)  # '1', '2', '3'
```

#### **`while` Loop**
```python
# Basic while loop
count = 0
while count < 3:
    print(count)  # 0, 1, 2
    count += 1
```

---

### **2. Advanced Looping Techniques**
#### **`enumerate()` for Index and Value**
```python
my_list = [10, 20, 30]
for idx, value in enumerate(my_list):
    print(f"Index: {idx}, Value: {value}")
# Output:
# Index: 0, Value: 10
# Index: 1, Value: 20
# Index: 2, Value: 30
```

#### **`zip()` for Parallel Iteration**
```python
names = ["Alice", "Bob", "Charlie"]
ages = [25, 30, 35]
for name, age in zip(names, ages):
    print(f"{name} is {age} years old")
# Output:
# Alice is 25 years old
# Bob is 30 years old
# Charlie is 35 years old
```

#### **`items()` for Dictionary Iteration**
```python
my_dict = {'a': 1, 'b': 2, 'c': 3}
for key, value in my_dict.items():
    print(f"Key: {key}, Value: {value}")
# Output:
# Key: a, Value: 1
# Key: b, Value: 2
# Key: c, Value: 3
```

#### **`reversed()` for Reverse Iteration**
```python
for num in reversed(range(1, 4)):
    print(num)  # 3, 2, 1
```

#### **`sorted()` for Sorted Iteration**
```python
my_list = [3, 1, 4, 1, 5]
for num in sorted(my_list):
    print(num)  # 1, 1, 3, 4, 5
```

---

### **3. Loop Control Statements**
#### **`break`**
- Exits the loop immediately.

```python
for num in range(10):
    if num == 5:
        break
    print(num)  # 0, 1, 2, 3, 4
```

#### **`continue`**
- Skips the rest of the loop body and continues to the next iteration.

```python
for num in range(5):
    if num == 2:
        continue
    print(num)  # 0, 1, 3, 4
```

#### **`else` with Loops**
- The `else` block executes if the loop completes without hitting a `break`.

```python
for num in range(3):
    print(num)
else:
    print("Loop completed")  # 0, 1, 2, Loop completed
```

---

### **4. Nested Loops**
```python
# Nested for loop
for i in range(3):
    for j in range(2):
        print(f"i={i}, j={j}")
# Output:
# i=0, j=0
# i=0, j=1
# i=1, j=0
# i=1, j=1
# i=2, j=0
# i=2, j=1
```

---

### **5. Real-World Use Cases**
#### **Processing Nested Data**
```python
# Iterate over a list of tuples
my_list2 = [(1, 2), (3, 4), (5, 6)]
for a, b in my_list2:
    print(f"Sum: {a + b}")
# Output:
# Sum: 3
# Sum: 7
# Sum: 11
```

#### **User Input Validation**
```python
# Wait until user quits
msg = ''
while msg != 'quit':
    msg = input("What should I do? ")
    print(msg)
```

#### **Filtering Data**
```python
# Filter even numbers
numbers = [1, 2, 3, 4, 5, 6]
evens = [num for num in numbers if num % 2 == 0]
print(evens)  # [2, 4, 6]
```

---

### **6. Advanced Techniques**
#### **Using `itertools` for Advanced Iteration**
```python
import itertools

# Infinite loop with itertools.count
for i in itertools.count(start=1, step=2):
    if i > 10:
        break
    print(i)  # 1, 3, 5, 7, 9

# Permutations
for perm in itertools.permutations([1, 2, 3]):
    print(perm)
# Output:
# (1, 2, 3)
# (1, 3, 2)
# (2, 1, 3)
# (2, 3, 1)
# (3, 1, 2)
# (3, 2, 1)
```

#### **Using `generators` for Lazy Evaluation**
```python
# Generator function
def generate_numbers():
    for i in range(3):
        yield i

for num in generate_numbers():
    print(num)  # 0, 1, 2
```

---

### **7. Performance Considerations**
#### **Avoid Unnecessary Computations in Loops**
```python
# Bad practice
for i in range(len(my_list)):
    print(my_list[i])

# Good practice
for item in my_list:
    print(item)
```

#### **Use List Comprehensions for Simple Transformations**
```python
# Bad practice
squares = []
for num in range(10):
    squares.append(num**2)

# Good practice
squares = [num**2 for num in range(10)]
```

---

### **8. Common Pitfalls**
#### **Infinite Loops**
- Ensure the loop condition will eventually become `False`.

```python
# Bad practice
# while True:
#     print("Infinite loop")

# Good practice
count = 0
while count < 5:
    print(count)
    count += 1
```

#### **Modifying a List While Iterating**
- Avoid modifying a list while iterating over it.

```python
# Bad practice
my_list = [1, 2, 3, 4]
for num in my_list:
    if num == 2:
        my_list.remove(num)  # Unexpected behavior

# Good practice
my_list = [num for num in my_list if num != 2]
```

---

### **9. Summary of Key Points**
- Use `for` loops for **iterating over sequences**.
- Use `while` loops for **condition-based iteration**.
- Leverage `enumerate()`, `zip()`, and `items()` for **advanced iteration**.
- Use `break`, `continue`, and `else` for **loop control**.
- Avoid **infinite loops** and **modifying lists during iteration**.

---


Range
-----
```python
range(10)          # range(0, 10) --> 0 to 9
range(1,10)        # range(1, 10)
list(range(0,10,2))# [0, 2, 4, 6, 8]
```

---

### **1. Basics of `range`**
```python
# Create a range from 0 to 9
r = range(10)
print(list(r))  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# Create a range from 1 to 9
r = range(1, 10)
print(list(r))  # [1, 2, 3, 4, 5, 6, 7, 8, 9]

# Create a range from 0 to 8 with a step of 2
r = range(0, 10, 2)
print(list(r))  # [0, 2, 4, 6, 8]
```

---

### **2. Advanced Range Techniques**
#### **Negative Steps**
```python
# Create a range from 10 to 1 with a step of -1
r = range(10, 0, -1)
print(list(r))  # [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]

# Create a range from 10 to 0 with a step of -2
r = range(10, 0, -2)
print(list(r))  # [10, 8, 6, 4, 2]
```

#### **Using `range` with `len`**
```python
my_list = ['a', 'b', 'c', 'd']
for i in range(len(my_list)):
    print(f"Index: {i}, Value: {my_list[i]}")
# Output:
# Index: 0, Value: a
# Index: 1, Value: b
# Index: 2, Value: c
# Index: 3, Value: d
```

#### **Using `range` with `enumerate`**
```python
my_list = ['a', 'b', 'c', 'd']
for i, value in enumerate(my_list):
    print(f"Index: {i}, Value: {value}")
# Output:
# Index: 0, Value: a
# Index: 1, Value: b
# Index: 2, Value: c
# Index: 3, Value: d
```

---

### **3. Real-World Use Cases**
#### **Generating Indices for Iteration**
```python
# Iterate over a list with indices
my_list = ['apple', 'banana', 'cherry']
for i in range(len(my_list)):
    print(f"Item {i + 1}: {my_list[i]}")
# Output:
# Item 1: apple
# Item 2: banana
# Item 3: cherry
```

#### **Creating Custom Sequences**
```python
# Generate a list of even numbers
evens = list(range(0, 20, 2))
print(evens)  # [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]

# Generate a list of odd numbers
odds = list(range(1, 20, 2))
print(odds)  # [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
```

#### **Simulating a Countdown**
```python
# Countdown from 10 to 1
for i in range(10, 0, -1):
    print(i)
print("Blast off!")
# Output:
# 10
# 9
# 8
# 7
# 6
# 5
# 4
# 3
# 2
# 1
# Blast off!
```

---

### **4. Performance Considerations**
#### **Memory Efficiency**
- `range` is **memory-efficient** because it generates numbers on-the-fly instead of storing them in a list.

```python
# Memory-efficient iteration
for i in range(1000000):
    pass  # No memory issues
```

#### **Avoid Converting `range` to List Unnecessarily**
- Converting `range` to a list consumes memory.

```python
# Bad practice
numbers = list(range(1000000))  # Consumes memory

# Good practice
for i in range(1000000):
    pass  # Memory-efficient
```

---

### **5. Advanced Techniques**
#### **Using `range` with `zip`**
```python
# Combine range with zip for parallel iteration
names = ['Alice', 'Bob', 'Charlie']
ages = [25, 30, 35]
for i, (name, age) in enumerate(zip(names, ages)):
    print(f"{i}: {name} is {age} years old")
# Output:
# 0: Alice is 25 years old
# 1: Bob is 30 years old
# 2: Charlie is 35 years old
```

#### **Using `range` with `itertools`**
```python
import itertools

# Infinite range with itertools.count
for i in itertools.count(start=1, step=2):
    if i > 10:
        break
    print(i)  # 1, 3, 5, 7, 9
```

#### **Custom Range-Like Functions**
```python
# Create a custom range-like function
def custom_range(start, stop, step=1):
    while start < stop:
        yield start
        start += step

for i in custom_range(0, 10, 2):
    print(i)  # 0, 2, 4, 6, 8
```

---

### **6. Common Pitfalls**
#### **Off-by-One Errors**
- Ensure the `stop` value is correct to avoid missing the last element.

```python
# Bad practice (misses 10)
for i in range(1, 10):
    print(i)  # 1 to 9

# Good practice (includes 10)
for i in range(1, 11):
    print(i)  # 1 to 10
```

#### **Negative Steps with Incorrect Bounds**
- Ensure the `start` and `stop` values are correct when using negative steps.

```python
# Bad practice (no output)
for i in range(10, 0, 1):
    print(i)  # No output

# Good practice
for i in range(10, 0, -1):
    print(i)  # 10 to 1
```

---

### **7. Summary of Key Points**
- Use `range` for **efficient iteration** over sequences of numbers.
- Leverage **negative steps** for reverse iteration.
- Avoid converting `range` to a list unless necessary.
- Use `range` with `enumerate`, `zip`, and `itertools` for **advanced iteration**.
- Be mindful of **off-by-one errors** and **incorrect bounds**.

---


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
------

### **1. Basics of `enumerate`**
```python
# Enumerate over a list
my_list = ['a', 'b', 'c']
for i, value in enumerate(my_list):
    print(f"Index: {i}, Value: {value}")
# Output:
# Index: 0, Value: a
# Index: 1, Value: b
# Index: 2, Value: c
```

#### **Custom Start Index**
```python
# Start enumeration from 1
for i, value in enumerate(my_list, start=1):
    print(f"Index: {i}, Value: {value}")
# Output:
# Index: 1, Value: a
# Index: 2, Value: b
# Index: 3, Value: c
```

---

### **2. Advanced Techniques**
#### **Enumerate with `zip`**
```python
# Combine enumerate with zip for parallel iteration
names = ['Alice', 'Bob', 'Charlie']
ages = [25, 30, 35]
for i, (name, age) in enumerate(zip(names, ages)):
    print(f"{i}: {name} is {age} years old")
# Output:
# 0: Alice is 25 years old
# 1: Bob is 30 years old
# 2: Charlie is 35 years old
```

#### **Enumerate with Dictionaries**
```python
# Enumerate over dictionary keys and values
my_dict = {'a': 1, 'b': 2, 'c': 3}
for i, (key, value) in enumerate(my_dict.items()):
    print(f"Index: {i}, Key: {key}, Value: {value}")
# Output:
# Index: 0, Key: a, Value: 1
# Index: 1, Key: b, Value: 2
# Index: 2, Key: c, Value: 3
```

#### **Enumerate with Nested Loops**
```python
# Enumerate over nested lists
nested_list = [[1, 2], [3, 4], [5, 6]]
for i, sublist in enumerate(nested_list):
    for j, value in enumerate(sublist):
        print(f"Row: {i}, Column: {j}, Value: {value}")
# Output:
# Row: 0, Column: 0, Value: 1
# Row: 0, Column: 1, Value: 2
# Row: 1, Column: 0, Value: 3
# Row: 1, Column: 1, Value: 4
# Row: 2, Column: 0, Value: 5
# Row: 2, Column: 1, Value: 6
```

---

### **3. Real-World Use Cases**
#### **Tracking Progress in a Loop**
```python
# Track progress with enumerate
tasks = ['task1', 'task2', 'task3', 'task4']
for i, task in enumerate(tasks, start=1):
    print(f"Processing task {i} of {len(tasks)}: {task}")
# Output:
# Processing task 1 of 4: task1
# Processing task 2 of 4: task2
# Processing task 3 of 4: task3
# Processing task 4 of 4: task4
```

#### **Generating Indexed Output**
```python
# Generate indexed output for a report
data = ['apple', 'banana', 'cherry']
for i, item in enumerate(data, start=1):
    print(f"{i}. {item}")
# Output:
# 1. apple
# 2. banana
# 3. cherry
```

#### **Debugging with Enumerate**
```python
# Debug a loop by printing indices and values
my_list = [10, 20, 30, 40]
for i, value in enumerate(my_list):
    if value == 30:
        print(f"Found 30 at index {i}")  # Found 30 at index 2
```

---

### **4. Performance Considerations**
#### **Memory Efficiency**
- `enumerate` is **memory-efficient** because it generates indices and values on-the-fly.

```python
# Memory-efficient iteration
for i, value in enumerate(range(1000000)):
    pass  # No memory issues
```

#### **Avoid Unnecessary Conversions**
- Avoid converting `enumerate` to a list unless necessary.

```python
# Bad practice
enumerated_list = list(enumerate(my_list))  # Consumes memory

# Good practice
for i, value in enumerate(my_list):
    pass  # Memory-efficient
```

---

### **5. Advanced Techniques**
#### **Using `enumerate` with `itertools`**
```python
import itertools

# Enumerate over an infinite iterator
for i, value in enumerate(itertools.count(start=1, step=2)):
    if i >= 5:
        break
    print(f"Index: {i}, Value: {value}")
# Output:
# Index: 0, Value: 1
# Index: 1, Value: 3
# Index: 2, Value: 5
# Index: 3, Value: 7
# Index: 4, Value: 9
```

#### **Custom Enumerate-Like Functions**
```python
# Create a custom enumerate-like function
def custom_enumerate(iterable, start=0):
    for value in iterable:
        yield start, value
        start += 1

for i, value in custom_enumerate(['a', 'b', 'c'], start=1):
    print(f"Index: {i}, Value: {value}")
# Output:
# Index: 1, Value: a
# Index: 2, Value: b
# Index: 3, Value: c
```

---

### **6. Common Pitfalls**
#### **Off-by-One Errors**
- Ensure the `start` value is correct to avoid off-by-one errors.

```python
# Bad practice (starts from 0 by default)
for i, value in enumerate(['a', 'b', 'c']):
    print(f"Task {i + 1}: {value}")  # Task 1: a, Task 2: b, Task 3: c

# Good practice (explicit start)
for i, value in enumerate(['a', 'b', 'c'], start=1):
    print(f"Task {i}: {value}")  # Task 1: a, Task 2: b, Task 3: c
```

#### **Ignoring the Index**
- If you don’t need the index, use `_` as a placeholder.

```python
# Bad practice
for i, value in enumerate(['a', 'b', 'c']):
    print(value)  # Index is unused

# Good practice
for _, value in enumerate(['a', 'b', 'c']):
    print(value)
```

---

### **7. Summary of Key Points**
- Use `enumerate` for **efficient iteration with indices**.
- Leverage the `start` parameter to **customize the starting index**.
- Combine `enumerate` with `zip`, `itertools`, and dictionaries for **advanced iteration**.
- Be mindful of **off-by-one errors** and **unnecessary conversions**.

---


The `enumerate` function in Python is a powerful tool for iterating over sequences (like lists, strings, or tuples) while keeping track of both the **index** and the **element**. To make your example more advanced, let's explore some creative and practical ways to use `enumerate` in real-world scenarios.

---

### **1. Enumerate with a Custom Start Index**
By default, `enumerate` starts indexing at `0`. You can specify a custom start index using the `start` parameter.

```python
for i, el in enumerate('helloo', start=1):
    print(f'{i}, {el}')
```

**Output:**
```
1, h
2, e
3, l
4, l
5, o
6, o
```

---

### **2. Enumerate with Conditional Logic**
You can use `enumerate` to filter or process elements based on their index.

```python
for i, el in enumerate('helloo'):
    if i % 2 == 0:  # Only print elements at even indices
        print(f'{i}, {el}')
```

**Output:**
```
0, h
2, l
4, o
```

---

### **3. Enumerate with Multiple Sequences**
You can use `enumerate` with multiple sequences (e.g., lists) to compare or combine them.

```python
list1 = ['apple', 'banana', 'cherry']
list2 = [1.2, 3.4, 5.6]

for i, (fruit, price) in enumerate(zip(list1, list2)):
    print(f'{i}: {fruit} costs ${price:.2f}')
```

**Output:**
```
0: apple costs $1.20
1: banana costs $3.40
2: cherry costs $5.60
```

---

### **4. Enumerate with Dictionary Creation**
You can use `enumerate` to create a dictionary where the keys are indices and the values are elements.

```python
word = 'helloo'
index_dict = {i: el for i, el in enumerate(word)}
print(index_dict)
```

**Output:**
```
{0: 'h', 1: 'e', 2: 'l', 3: 'l', 4: 'o', 5: 'o'}
```

---

### **5. Enumerate with Nested Loops**
You can use `enumerate` in nested loops to track indices at multiple levels.

```python
matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

for i, row in enumerate(matrix):
    for j, value in enumerate(row):
        print(f'Element at ({i}, {j}) is {value}')
```

**Output:**
```
Element at (0, 0) is 1
Element at (0, 1) is 2
Element at (0, 2) is 3
Element at (1, 0) is 4
Element at (1, 1) is 5
Element at (1, 2) is 6
Element at (2, 0) is 7
Element at (2, 1) is 8
Element at (2, 2) is 9
```

---

### **6. Enumerate with File Handling**
You can use `enumerate` to read lines from a file while keeping track of line numbers.

```python
with open('example.txt', 'r') as file:
    for line_num, line in enumerate(file, start=1):
        print(f'Line {line_num}: {line.strip()}')
```

**Output (for `example.txt`):**
```
Line 1: This is line 1.
Line 2: This is line 2.
Line 3: This is line 3.
```

---

### **7. Enumerate with List Comprehension**
You can use `enumerate` in list comprehensions to create new lists based on indices and elements.

```python
word = 'helloo'
uppercase_indices = [i for i, el in enumerate(word) if el.isupper()]
print(uppercase_indices)
```

**Output:**
```
[]  # No uppercase letters in 'helloo'
```

---

### **8. Enumerate with Custom Objects**
You can use `enumerate` with custom objects or classes.

```python
class Fruit:
    def __init__(self, name, color):
        self.name = name
        self.color = color

fruits = [Fruit('apple', 'red'), Fruit('banana', 'yellow'), Fruit('cherry', 'red')]

for i, fruit in enumerate(fruits):
    print(f'{i}: {fruit.name} is {fruit.color}')
```

**Output:**
```
0: apple is red
1: banana is yellow
2: cherry is red
```

---

### **9. Enumerate with Error Handling**
You can use `enumerate` in combination with error handling to process problematic data.

```python
data = ['1', '2', 'three', '4']

for i, el in enumerate(data):
    try:
        value = int(el)
        print(f'{i}: {value} is a valid integer')
    except ValueError:
        print(f'{i}: "{el}" is not a valid integer')
```

**Output:**
```
0: 1 is a valid integer
1: 2 is a valid integer
2: "three" is not a valid integer
3: 4 is a valid integer
```

---

### **10. Enumerate with Advanced Formatting**
You can use `enumerate` with advanced string formatting for better readability.

```python
word = 'helloo'
for i, el in enumerate(word):
    print(f'Index: {i:>2}, Element: {el}')
```

**Output:**
```
Index:  0, Element: h
Index:  1, Element: e
Index:  2, Element: l
Index:  3, Element: l
Index:  4, Element: o
Index:  5, Element: o
```

---

### **Summary**
- `enumerate` is versatile and can be used in many advanced scenarios.
- Use it with custom start indices, conditional logic, dictionaries, file handling, and more.
- Combine it with other Python features like list comprehensions, error handling, and string formatting for powerful results.

Counter
-----
```python
from collections import Counter
colors = ['red', 'blue', 'yellow', 'blue', 'red', 'blue']
counter = Counter(colors)# Counter({'blue': 3, 'red': 2, 'yellow': 1})
counter.most_common()[0] # ('blue', 3)
```

The `collections.Counter` class in Python is a powerful tool for counting hashable objects. It is a subclass of `dict` and provides a convenient way to count occurrences of elements in a collection.
---

### **1. Basic Usage**
Count occurrences of elements in a list.

```python
from collections import Counter

colors = ['red', 'blue', 'yellow', 'blue', 'red', 'blue']
counter = Counter(colors)
print(counter)
```

**Output:**
```
Counter({'blue': 3, 'red': 2, 'yellow': 1})
```

---

### **2. Most Common Elements**
Use `most_common()` to retrieve the most frequent elements.

```python
print(counter.most_common(2))  # Top 2 most common elements
```

**Output:**
```
[('blue', 3), ('red', 2)]
```

---

### **3. Update Counter**
You can update a `Counter` with new data using the `update()` method.

```python
counter.update(['red', 'green', 'blue'])
print(counter)
```

**Output:**
```
Counter({'blue': 4, 'red': 3, 'yellow': 1, 'green': 1})
```

---

### **4. Subtract Counts**
Use `subtract()` to reduce counts based on another iterable or `Counter`.

```python
counter.subtract(['red', 'blue'])
print(counter)
```

**Output:**
```
Counter({'blue': 3, 'red': 2, 'yellow': 1, 'green': 1})
```

---

### **5. Combine Counters**
You can combine two `Counter` objects using arithmetic operations.

```python
counter1 = Counter(['red', 'blue', 'yellow'])
counter2 = Counter(['blue', 'green'])

# Addition
combined = counter1 + counter2
print(combined)

# Subtraction
difference = counter1 - counter2
print(difference)
```

**Output:**
```
Counter({'blue': 2, 'red': 1, 'yellow': 1, 'green': 1})
Counter({'red': 1, 'yellow': 1})
```

---

### **6. Intersection and Union**
Use `&` for intersection (minimum counts) and `|` for union (maximum counts).

```python
intersection = counter1 & counter2
union = counter1 | counter2

print(intersection)
print(union)
```

**Output:**
```
Counter({'blue': 1})
Counter({'blue': 1, 'red': 1, 'yellow': 1, 'green': 1})
```

---

### **7. Total Count**
Use the `total()` method (Python 3.10+) to get the total number of elements counted.

```python
print(counter.total())  # Total count of all elements
```

**Output:**
```
7
```

---

### **8. Convert to Dictionary**
Convert a `Counter` to a regular dictionary.

```python
counter_dict = dict(counter)
print(counter_dict)
```

**Output:**
```
{'blue': 3, 'red': 2, 'yellow': 1, 'green': 1}
```

---

### **9. Count Characters in a String**
Use `Counter` to count characters in a string.

```python
text = "hello world"
char_counter = Counter(text)
print(char_counter)
```

**Output:**
```
Counter({'l': 3, 'o': 2, 'h': 1, 'e': 1, ' ': 1, 'w': 1, 'r': 1, 'd': 1})
```

---

### **10. Count Words in a Text**
Use `Counter` to count words in a text.

```python
text = "hello world hello python world"
words = text.split()
word_counter = Counter(words)
print(word_counter)
```

**Output:**
```
Counter({'hello': 2, 'world': 2, 'python': 1})
```

---

### **11. Count Elements in a List of Tuples**
Use `Counter` to count elements in a list of tuples.

```python
data = [('apple', 'red'), ('banana', 'yellow'), ('apple', 'green')]
counter = Counter(color for _, color in data)
print(counter)
```

**Output:**
```
Counter({'red': 1, 'yellow': 1, 'green': 1})
```

---

### **12. Count Elements in a Nested List**
Use `Counter` with a flattened nested list.

```python
from itertools import chain

nested_list = [[1, 2, 3], [2, 3, 4], [3, 4, 5]]
flattened = list(chain.from_iterable(nested_list))
counter = Counter(flattened)
print(counter)
```

**Output:**
```
Counter({3: 3, 2: 2, 4: 2, 1: 1, 5: 1})
```

---

### **13. Count Unique Elements**
Use `Counter` to count unique elements in a list.

```python
unique_counter = Counter(set(colors))
print(unique_counter)
```

**Output:**
```
Counter({'blue': 1, 'red': 1, 'yellow': 1})
```

---

### **14. Count with Custom Objects**
Use `Counter` with custom objects (requires objects to be hashable).

```python
class Fruit:
    def __init__(self, name):
        self.name = name

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self.name == other.name

fruits = [Fruit('apple'), Fruit('banana'), Fruit('apple')]
counter = Counter(fruits)
print(counter)
```

**Output:**
```
Counter({<Fruit: apple>: 2, <Fruit: banana>: 1})
```

---

### **15. Count with Default Values**
Use `defaultdict` with `Counter` for default values.

```python
from collections import defaultdict

default_counter = defaultdict(int, counter)
print(default_counter['purple'])  # Returns 0 (default value)
```

**Output:**
```
0
```

---

### **16. Count with Sorting**
Sort a `Counter` by keys or values.

```python
# Sort by keys
sorted_by_key = sorted(counter.items(), key=lambda x: x[0])
print(sorted_by_key)

# Sort by values
sorted_by_value = sorted(counter.items(), key=lambda x: x[1])
print(sorted_by_value)
```

**Output:**
```
[('blue', 3), ('green', 1), ('red', 2), ('yellow', 1)]
[('green', 1), ('yellow', 1), ('red', 2), ('blue', 3)]
```

---

### **17. Count with Zero or Negative Values**
`Counter` can handle zero or negative counts.

```python
counter['purple'] = 0
counter['blue'] -= 5
print(counter)
```

**Output:**
```
Counter({'red': 2, 'yellow': 1, 'green': 1, 'purple': 0, 'blue': -2})
```

---

### **18. Count with Large Datasets**
Use `Counter` with large datasets efficiently.

```python
import random

large_data = [random.randint(1, 100) for _ in range(1000000)]
counter = Counter(large_data)
print(counter.most_common(5))  # Top 5 most common numbers
```

**Output:**
```
[(42, 10123), (17, 10098), (89, 10095), (56, 10091), (23, 10089)]
```

---

### **Summary**
- `Counter` is a versatile tool for counting hashable objects.
- Use it for counting elements in lists, strings, or custom objects.
- Combine it with other Python features like `most_common()`, arithmetic operations, and sorting for advanced use cases.



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


`namedtuple` is a powerful and memory-efficient way to create **immutable**, **lightweight objects** with named fields. It is part of the `collections` module and is often used as a replacement for simple classes. Below is an **advanced cheatsheet** for using `namedtuple`, covering various use cases and advanced features.

---

### **1. Basic Usage**
Create a `namedtuple` with named fields.

```python
from collections import namedtuple

# Define a namedtuple
Point = namedtuple('Point', ['x', 'y'])

# Create an instance
p = Point(1, y=2)
print(p)
```

**Output:**
```
Point(x=1, y=2)
```

---

### **2. Accessing Fields**
Access fields using dot notation or indexing.

```python
print(p.x)       # 1
print(p[0])      # 1
print(getattr(p, 'y'))  # 2
```

---

### **3. Immutable Nature**
`namedtuple` instances are immutable. Attempting to modify a field will raise an error.

```python
try:
    p.x = 3  # Attempt to modify
except AttributeError as e:
    print(f"Error: {e}")
```

**Output:**
```
Error: can't set attribute
```

---

### **4. Convert to Dictionary**
Convert a `namedtuple` to a dictionary using `_asdict()`.

```python
print(p._asdict())
```

**Output:**
```
{'x': 1, 'y': 2}
```

---

### **5. Replace Fields**
Use `_replace()` to create a new instance with updated fields.

```python
p_new = p._replace(x=10)
print(p_new)
```

**Output:**
```
Point(x=10, y=2)
```

---

### **6. Default Values**
You can provide default values for fields using a dictionary.

```python
Point = namedtuple('Point', ['x', 'y'], defaults=[0, 0])
p_default = Point()
print(p_default)
```

**Output:**
```
Point(x=0, y=0)
```

---

### **7. Field Names**
Retrieve the field names using `_fields`.

```python
print(p._fields)
```

**Output:**
```
('x', 'y')
```

---

### **8. Create from Iterable**
Create a `namedtuple` from an iterable using `_make()`.

```python
p_from_iter = Point._make([3, 4])
print(p_from_iter)
```

**Output:**
```
Point(x=3, y=4)
```

---

### **9. Nested Named Tuples**
You can nest `namedtuple` instances.

```python
Person = namedtuple('Person', ['name', 'height'])
Address = namedtuple('Address', ['city', 'zipcode'])

person = Person('Jean-Luc', 187)
address = Address('Paris', 75001)

person_with_address = namedtuple('PersonWithAddress', ['person', 'address'])
combined = person_with_address(person, address)
print(combined)
```

**Output:**
```
PersonWithAddress(person=Person(name='Jean-Luc', height=187), address=Address(city='Paris', zipcode=75001))
```

---

### **10. Type Annotations**
Add type hints to `namedtuple` fields using `typing.NamedTuple`.

```python
from typing import NamedTuple

class Point(NamedTuple):
    x: int
    y: int

p = Point(1, 2)
print(p)
```

**Output:**
```
Point(x=1, y=2)
```

---

### **11. Custom Methods**
Add custom methods to a `namedtuple` by subclassing.

```python
class Point(namedtuple('Point', ['x', 'y'])):
    def distance(self, other):
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5

p1 = Point(1, 2)
p2 = Point(4, 6)
print(p1.distance(p2))
```

**Output:**
```
5.0
```

---

### **12. Serialization**
`namedtuple` instances can be serialized using `pickle`.

```python
import pickle

# Serialize
serialized = pickle.dumps(p)
print(serialized)

# Deserialize
deserialized = pickle.loads(serialized)
print(deserialized)
```

**Output:**
```
b'\x80\x04\x95\x1b\x00\x00\x00\x00\x00\x00\x00\x8c\x08__main__\x94\x8c\x05Point\x94\x93\x94K\x01K\x02\x86\x94.'
Point(x=1, y=2)
```

---

### **13. JSON Serialization**
Convert a `namedtuple` to JSON using `_asdict()`.

```python
import json

p_dict = p._asdict()
p_json = json.dumps(p_dict)
print(p_json)
```

**Output:**
```
{"x": 1, "y": 2}
```

---

### **14. Use in Data Processing**
`namedtuple` is useful for processing structured data.

```python
data = [
    ('Alice', 25),
    ('Bob', 30),
    ('Charlie', 35)
]

Person = namedtuple('Person', ['name', 'age'])
people = [Person._make(row) for row in data]

for person in people:
    print(f'{person.name} is {person.age} years old')
```

**Output:**
```
Alice is 25 years old
Bob is 30 years old
Charlie is 35 years old
```

---

### **15. Use in Databases**
`namedtuple` can represent rows from a database query.

```python
import sqlite3

# Example database
conn = sqlite3.connect(':memory:')
conn.execute('CREATE TABLE users (name TEXT, age INTEGER)')
conn.execute('INSERT INTO users VALUES ("Alice", 25), ("Bob", 30)')

# Query and convert to namedtuple
cursor = conn.execute('SELECT name, age FROM users')
User = namedtuple('User', ['name', 'age'])
users = [User._make(row) for row in cursor]

for user in users:
    print(user)
```

**Output:**
```
User(name='Alice', age=25)
User(name='Bob', age=30)
```

---

### **16. Use in CSV Processing**
`namedtuple` can represent rows from a CSV file.

```python
import csv

# Example CSV data
csv_data = '''name,age
Alice,25
Bob,30
Charlie,35'''

# Read CSV and convert to namedtuple
reader = csv.reader(csv_data.splitlines())
header = next(reader)
Person = namedtuple('Person', header)
people = [Person._make(row) for row in reader]

for person in people:
    print(person)
```

**Output:**
```
Person(name='Alice', age='25')
Person(name='Bob', age='30')
Person(name='Charlie', age='35')
```

---

### **Summary**
- `namedtuple` is a lightweight, immutable, and memory-efficient way to create structured data.
- Use it for representing records, database rows, CSV data, or any structured information.
- Combine it with other Python features like type hints, serialization, and custom methods for advanced use cases.



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


---

### **1. Basics of `OrderedDict`**
```python
from collections import OrderedDict

# Create an OrderedDict
programmers = OrderedDict()
programmers['Tim'] = ['python', 'javascript']
programmers['Sarah'] = ['C++']
programmers['Bia'] = ['Ruby', 'Python', 'Go']

# Iterate over the OrderedDict
for name, langs in programmers.items():
    print(name + '-->')
    for lang in langs:
        print('\t' + lang)
# Output:
# Tim-->
#     python
#     javascript
# Sarah-->
#     C++
# Bia-->
#     Ruby
#     Python
#     Go
```

---

### **2. Advanced Techniques**
#### **Reordering Items**
```python
# Move an item to the end
programmers.move_to_end('Tim')
print(programmers)  # OrderedDict([('Sarah', ['C++']), ('Bia', ['Ruby', 'Python', 'Go']), ('Tim', ['python', 'javascript'])])

# Move an item to the beginning
programmers.move_to_end('Bia', last=False)
print(programmers)  # OrderedDict([('Bia', ['Ruby', 'Python', 'Go']), ('Sarah', ['C++']), ('Tim', ['python', 'javascript'])])
```

#### **Pop Items**
```python
# Pop the last item
last_item = programmers.popitem()
print(last_item)  # ('Tim', ['python', 'javascript'])

# Pop a specific item
sarah_langs = programmers.pop('Sarah')
print(sarah_langs)  # ['C++']
```

#### **Update and Maintain Order**
```python
# Update an existing item (order is preserved)
programmers['Bia'].append('Java')
print(programmers)  # OrderedDict([('Bia', ['Ruby', 'Python', 'Go', 'Java']), ('Tim', ['python', 'javascript'])])
```

---

### **3. Real-World Use Cases**
#### **Maintaining Insertion Order in Configurations**
```python
# Store configuration settings in order
config = OrderedDict()
config['host'] = 'localhost'
config['port'] = 8080
config['timeout'] = 30

for key, value in config.items():
    print(f"{key}: {value}")
# Output:
# host: localhost
# port: 8080
# timeout: 30
```

#### **Tracking History of Operations**
```python
# Track the order of operations
operations = OrderedDict()
operations['login'] = 'success'
operations['upload'] = 'success'
operations['logout'] = 'success'

for operation, status in operations.items():
    print(f"{operation}: {status}")
# Output:
# login: success
# upload: success
# logout: success
```

#### **Implementing LRU Cache**
```python
from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

# Example usage
cache = LRUCache(2)
cache.put(1, 1)
cache.put(2, 2)
print(cache.get(1))  # 1
cache.put(3, 3)
print(cache.get(2))  # -1
```

---

### **4. Performance Considerations**
#### **Memory Overhead**
- `OrderedDict` uses more memory than a regular `dict` because it maintains a doubly-linked list to preserve order.

```python
import sys
regular_dict = {'a': 1, 'b': 2, 'c': 3}
ordered_dict = OrderedDict([('a', 1), ('b', 2), ('c', 3)])

print(sys.getsizeof(regular_dict))  # Less memory
print(sys.getsizeof(ordered_dict))  # More memory
```

#### **Time Complexity**
- Most operations (insertion, deletion, lookup) in `OrderedDict` have the same time complexity as a regular `dict` (O(1)), but reordering operations like `move_to_end` are also O(1).

---

### **5. Advanced Techniques**
#### **Combining `OrderedDict` with `defaultdict`**
```python
from collections import OrderedDict, defaultdict

# Create an OrderedDict with defaultdict
programmers = OrderedDict()
programmers['Tim'] = defaultdict(list)
programmers['Tim']['languages'].append('python')
programmers['Tim']['languages'].append('javascript')

for name, data in programmers.items():
    print(name + '-->')
    for key, value in data.items():
        print(f"\t{key}: {', '.join(value)}")
# Output:
# Tim-->
#     languages: python, javascript
```

#### **Serializing `OrderedDict` to JSON**
```python
import json

# Convert OrderedDict to JSON
config = OrderedDict([('host', 'localhost'), ('port', 8080), ('timeout', 30)])
config_json = json.dumps(config)
print(config_json)  # {"host": "localhost", "port": 8080, "timeout": 30}
```

---

### **6. Common Pitfalls**
#### **Assuming Order in Regular `dict`**
- In Python 3.7+, regular `dict` maintains insertion order, but this is an implementation detail. Use `OrderedDict` if order preservation is critical.

```python
# Bad practice (assuming order in regular dict)
regular_dict = {'a': 1, 'b': 2, 'c': 3}
for key in regular_dict:
    print(key)  # May not preserve order in older Python versions

# Good practice (use OrderedDict)
ordered_dict = OrderedDict([('a', 1), ('b', 2), ('c', 3)])
for key in ordered_dict:
    print(key)  # Guaranteed order
```

#### **Overusing `OrderedDict`**
- Use `OrderedDict` only when order preservation is necessary. For most cases, a regular `dict` is sufficient.

---

### **7. Summary of Key Points**
- Use `OrderedDict` to **maintain insertion order**.
- Leverage `move_to_end` and `popitem` for **reordering and removing items**.
- Combine `OrderedDict` with `defaultdict` and `json` for **advanced use cases**.
- Be mindful of **memory overhead** and **time complexity**.

---

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

The use of `*args` and `**kwargs` in Python is a powerful feature that allows functions to accept a variable number of arguments. Below is an **advanced cheatsheet** that covers various use cases, best practices, and advanced techniques for working with `*args` and `**kwargs`.

---

### **1. Basic Usage**
#### **`*args`**
- Collects **positional arguments** into a tuple.
- Useful when the number of arguments is unknown.

```python
def add(*args):
    return sum(args)

print(add(1, 2, 3))  # 6
```

#### **`**kwargs`**
- Collects **keyword arguments** into a dictionary.
- Useful when the number of keyword arguments is unknown.

```python
def greet(**kwargs):
    for key, value in kwargs.items():
        print(f'{key}: {value}')

greet(name='Alice', age=25)
```

**Output:**
```
name: Alice
age: 25
```

---

### **2. Combining `*args` and `**kwargs`**
You can use both `*args` and `**kwargs` in the same function.

```python
def func(*args, **kwargs):
    print(f'Positional arguments: {args}')
    print(f'Keyword arguments: {kwargs}')

func(1, 2, x=3, y=4)
```

**Output:**
```
Positional arguments: (1, 2)
Keyword arguments: {'x': 3, 'y': 4}
```

---

### **3. Unpacking Arguments**
You can unpack collections into arguments using `*` and `**`.

```python
def func(a, b, c):
    print(a, b, c)

args = (1, 2, 3)
kwargs = {'a': 1, 'b': 2, 'c': 3}

func(*args)      # Equivalent to func(1, 2, 3)
func(**kwargs)   # Equivalent to func(a=1, b=2, c=3)
```

---

### **4. Advanced Argument Ordering**
The order of parameters in a function definition matters. Here are some advanced patterns:

#### **Pattern 1: Mandatory Keyword Arguments**
Use `*` to enforce keyword arguments after it.

```python
def func(a, b, *, c):
    print(a, b, c)

func(1, 2, c=3)  # Valid
func(1, 2, 3)    # Error: c must be a keyword argument
```

#### **Pattern 2: Mixed `*args` and `**kwargs`**
```python
def func(x, *args, y, **kwargs):
    print(f'x: {x}, args: {args}, y: {y}, kwargs: {kwargs}')

func(1, 2, 3, y=4, z=5)
```

**Output:**
```
x: 1, args: (2, 3), y: 4, kwargs: {'z': 5}
```

---

### **5. Unpacking in Data Structures**
You can use `*` and `**` to unpack collections into data structures.

#### **Unpacking Lists**
```python
combined_list = [*[1, 2, 3], *[4, 5]]
print(combined_list)  # [1, 2, 3, 4, 5]
```

#### **Unpacking Sets**
```python
combined_set = {*[1, 2, 3], *[3, 4, 5]}
print(combined_set)  # {1, 2, 3, 4, 5}
```

#### **Unpacking Dictionaries**
```python
combined_dict = {**{'a': 1, 'b': 2}, **{'c': 3}}
print(combined_dict)  # {'a': 1, 'b': 2, 'c': 3}
```

---

### **6. Extended Unpacking**
You can use `*` for extended unpacking in assignments.

```python
head, *body, tail = [1, 2, 3, 4, 5]
print(head)  # 1
print(body)  # [2, 3, 4]
print(tail)  # 5
```

---

### **7. Decorators with `*args` and `**kwargs`**
You can create decorators that work with any function signature using `*args` and `**kwargs`.

```python
def debug(func):
    def wrapper(*args, **kwargs):
        print(f'Calling {func.__name__} with args={args}, kwargs={kwargs}')
        result = func(*args, **kwargs)
        print(f'{func.__name__} returned {result}')
        return result
    return wrapper

@debug
def add(a, b):
    return a + b

add(1, 2)
```

**Output:**
```
Calling add with args=(1, 2), kwargs={}
add returned 3
```

---

### **8. Partial Function Application**
Use `functools.partial` to fix some arguments of a function.

```python
from functools import partial

def power(base, exponent):
    return base ** exponent

square = partial(power, exponent=2)
print(square(3))  # 9
```

---

### **9. Type Annotations with `*args` and `**kwargs`**
You can add type hints to `*args` and `**kwargs` using `typing`.

```python
from typing import Any

def func(*args: int, **kwargs: Any):
    print(f'args: {args}, kwargs: {kwargs}')

func(1, 2, x=3, y=4)
```

---

### **10. Advanced Use Cases**
#### **Dynamic Function Creation**
You can dynamically create functions with `*args` and `**kwargs`.

```python
def create_function(func_name):
    def dynamic_func(*args, **kwargs):
        print(f'{func_name} called with args={args}, kwargs={kwargs}')
    return dynamic_func

my_func = create_function('my_func')
my_func(1, 2, x=3)
```

**Output:**
```
my_func called with args=(1, 2), kwargs={'x': 3}
```

#### **Chaining Functions**
You can chain functions using `*args` and `**kwargs`.

```python
def add(a, b):
    return a + b

def multiply(a, b):
    return a * b

def chain_functions(func1, func2, *args, **kwargs):
    result1 = func1(*args, **kwargs)
    result2 = func2(result1, *args, **kwargs)
    return result2

print(chain_functions(add, multiply, 2, 3))  # (2 + 3) * 2 = 10
```

---

### **11. Best Practices**
1. **Use Descriptive Names**:
   - Use meaningful names like `*values` or `**options` instead of `*args` and `**kwargs` when possible.

2. **Document Your Functions**:
   - Clearly document the expected arguments and their purpose.

3. **Avoid Overusing `*args` and `**kwargs`**:
   - Overusing them can make your code harder to understand and debug.

4. **Combine with Type Hints**:
   - Use type hints to make your code more readable and maintainable.

---

### **Summary**
- `*args` collects positional arguments into a tuple.
- `**kwargs` collects keyword arguments into a dictionary.
- Use them for flexible function definitions, unpacking, decorators, and more.
- Combine with type hints and best practices for clean and maintainable code.

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

---

#### **1. Multiple Arguments**
```python
# Multi-argument lambda
operation = lambda a, b, c: (a + b) * c
print(operation(2, 3, 4))  # (2+3)*4 = 20
```

#### **2. Conditional Logic**
```python
# Ternary operator in lambda
grade = lambda score: 'Pass' if score >= 60 else 'Fail'
print(grade(75))  # 'Pass'
```

#### **3. Higher-Order Functions**
```python
# Using lambda with map, filter, reduce
from functools import reduce

numbers = [1, 2, 3, 4, 5]
squared = list(map(lambda x: x**2, numbers))          # [1, 4, 9, 16, 25]
evens = list(filter(lambda x: x % 2 == 0, numbers))   # [2, 4]
product = reduce(lambda x, y: x * y, numbers)         # 120
```

#### **4. Sorting with Custom Keys**
```python
# Sort by string length
words = ['apple', 'banana', 'cherry', 'date']
sorted_words = sorted(words, key=lambda x: len(x))  # ['date', 'apple', 'banana', 'cherry']
```

#### **5. Recursive Lambda (Advanced)**
```python
# Recursive factorial (requires assignment)
fact = lambda n: 1 if n == 0 else n * fact(n-1)
print(fact(5))  # 120
```

#### **6. Closures and Captured Variables**
```python
# Lambda capturing external variables
def multiplier(n):
    return lambda x: x * n

double = multiplier(2)
print(double(5))  # 10
```

---

### **Comprehensions: Advanced Usage**

#### **1. Nested Comprehensions**
```python
# Matrix transposition
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
transposed = [[row[i] for row in matrix] for i in range(3)]
# [[1, 4, 7], [2, 5, 8], [3, 6, 9]]
```

#### **2. Conditional Logic**
```python
# Conditional in value and filter
numbers = [1, 2, 3, 4, 5]
result = [x**2 if x % 2 == 0 else x**3 for x in numbers if x > 2]
# [27, 16, 125]
```

#### **3. Dictionary Comprehensions**
```python
# Swap keys and values (unique values only)
original = {'a': 1, 'b': 2, 'c': 3}
swapped = {v: k for k, v in original.items()}  # {1: 'a', 2: 'b', 3: 'c'}
```

#### **4. Set Comprehensions**
```python
# Unique vowels in a string
sentence = "the quick brown fox jumps over the lazy dog"
vowels = {char for char in sentence if char in 'aeiou'}  # {'a', 'e', 'i', 'o', 'u'}
```

#### **5. Generator Expressions**
```python
# Memory-efficient iteration
gen = (x**2 for x in range(1000000) if x % 2 == 0)
print(next(gen))  # 0
print(next(gen))  # 4
```

#### **6. Multi-Iterable Comprehensions**
```python
# Cartesian product
colors = ['red', 'green']
sizes = ['S', 'M']
products = [(color, size) for color in colors for size in sizes]
# [('red', 'S'), ('red', 'M'), ('green', 'S'), ('green', 'M')]
```

#### **7. Walrus Operator (Python 3.8+)**
```python
# Assign and use variables in comprehensions
data = ["apple", "banana", "cherry"]
filtered = [word.upper() for s in data if (word := s.lower()) == 'apple']
# ['APPLE']
```

#### **8. Combining with zip()**
```python
# Pairwise operations
names = ['Alice', 'Bob']
scores = [85, 92]
combined = {name: score for name, score in zip(names, scores)}  # {'Alice': 85, 'Bob': 92}
```

#### **9. Flattening Nested Lists**
```python
# Flatten 2D list
matrix = [[1, 2], [3, 4], [5, 6]]
flat = [num for row in matrix for num in row]  # [1, 2, 3, 4, 5, 6]
```

#### **10. Asynchronous Comprehensions (Python 3.6+)**
```python
# Requires async environment
async def async_demo():
    return [i async for i in async_generator()]
```

---

### **Best Practices**
1. **Lambda**:
   - Use for simple, one-line operations.
   - Avoid complex logic (use `def` for multi-line functions).
   - Use type hints where possible.

2. **Comprehensions**:
   - Keep them readable; split complex comprehensions into loops.
   - Use generator expressions for large datasets.
   - Avoid nested comprehensions beyond 2 levels.

---

### **Performance Considerations**
- **Generator Expressions**: Memory-efficient for large datasets.
- **Set/Dict Comprehensions**: Faster than loops for membership checks.
- **Map vs Comprehension**: `map` can be faster for simple transformations.

---

### **Advanced Use Cases**
#### **1. Matrix Operations**
```python
# Matrix multiplication (3x3)
A = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
B = [[9, 8, 7], [6, 5, 4], [3, 2, 1]]
result = [[sum(a*b for a, b in zip(row, col)) for col in zip(*B)] for row in A]
# [[30, 24, 18], [84, 69, 54], [138, 114, 90]]
```

#### **2. Prime Number Generation**
```python
# Sieve of Eratosthenes using list comprehension
n = 30
primes = [x for x in range(2, n) if all(x % y != 0 for y in range(2, int(x**0.5)+1))]
# [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
```

---

Ternary Condition
-------
```python
# <expression_if_true> if <condition> else <expression_if_false>

[a if a else 'zero' for a in [0, 1, 0, 3]] # ['zero', 1, 'zero', 3]
```

---

### **1. Basics of Ternary Operator**
```python
# Syntax: <expression_if_true> if <condition> else <expression_if_false>

# Example
a = 10
b = 20
max_value = a if a > b else b
print(max_value)  # 20
```

---

### **2. Advanced Techniques**
#### **Nested Ternary Operators**
```python
# Nested ternary for multiple conditions
a = 10
b = 20
c = 30
max_value = a if a > b and a > c else (b if b > c else c)
print(max_value)  # 30
```

#### **Ternary with List Comprehensions**
```python
# Replace zeros with 'zero'
numbers = [0, 1, 0, 3]
result = ['zero' if num == 0 else num for num in numbers]
print(result)  # ['zero', 1, 'zero', 3]
```

#### **Ternary with Dictionary Comprehensions**
```python
# Create a dictionary with ternary logic
numbers = [1, 2, 3, 4]
result = {num: 'even' if num % 2 == 0 else 'odd' for num in numbers}
print(result)  # {1: 'odd', 2: 'even', 3: 'odd', 4: 'even'}
```

#### **Ternary with Function Calls**
```python
# Use ternary to decide which function to call
def greet():
    return "Hello!"

def farewell():
    return "Goodbye!"

is_greeting = True
message = greet() if is_greeting else farewell()
print(message)  # Hello!
```

---

### **3. Real-World Use Cases**
#### **Default Values**
```python
# Provide a default value if a variable is None
user_input = None
value = user_input if user_input is not None else "default"
print(value)  # default
```

#### **Conditional Formatting**
```python
# Format a string based on a condition
score = 85
result = "Pass" if score >= 60 else "Fail"
print(f"Result: {result}")  # Result: Pass
```

#### **Filtering Data**
```python
# Filter and transform data in a list comprehension
numbers = [1, 2, 3, 4, 5]
filtered = [num * 2 if num % 2 == 0 else num for num in numbers]
print(filtered)  # [1, 4, 3, 8, 5]
```

---

### **4. Performance Considerations**
#### **Avoid Complex Ternary Expressions**
- Keep ternary expressions simple to maintain readability.

```python
# Bad practice (hard to read)
result = (a if a > b else (b if b > c else (c if c > d else d))

# Good practice (use if-else for complex logic)
if a > b:
    result = a
elif b > c:
    result = b
elif c > d:
    result = c
else:
    result = d
```

#### **Short-Circuiting**
- Ternary operators **short-circuit**, meaning only the selected expression is evaluated.

```python
# Only the selected expression is evaluated
def expensive_operation():
    print("Expensive operation executed")
    return 100

a = 10
result = a if a > 5 else expensive_operation()
print(result)  # 10 (expensive_operation() is not called)
```

---

### **5. Advanced Techniques**
#### **Ternary with Lambda Functions**
```python
# Use ternary in a lambda function
func = lambda x: "positive" if x > 0 else ("zero" if x == 0 else "negative")
print(func(5))   # positive
print(func(0))   # zero
print(func(-5))  # negative
```

#### **Ternary with `map`**
```python
# Apply ternary logic to each element in a list
numbers = [1, 2, 3, 4]
result = list(map(lambda x: 'even' if x % 2 == 0 else 'odd', numbers))
print(result)  # ['odd', 'even', 'odd', 'even']
```

#### **Ternary with `filter`**
```python
# Filter and transform data using ternary
numbers = [1, 2, 3, 4, 5]
filtered = list(filter(lambda x: x > 2, map(lambda x: x * 2 if x % 2 == 0 else x, numbers)))
print(filtered)  # [1, 4, 3, 8, 5]
```

---

### **6. Common Pitfalls**
#### **Overusing Ternary Operators**
- Avoid using ternary operators for complex logic; use `if-else` instead.

```python
# Bad practice (hard to read)
result = (a if a > b else (b if b > c else (c if c > d else d)))

# Good practice (use if-else)
if a > b:
    result = a
elif b > c:
    result = b
elif c > d:
    result = c
else:
    result = d
```

#### **Confusing Order of Operations**
- Use parentheses to clarify the order of operations in nested ternary expressions.

```python
# Bad practice (confusing)
result = a if a > b else b if b > c else c

# Good practice (use parentheses)
result = (a if a > b else (b if b > c else c))
```

---

### **7. Summary of Key Points**
- Use ternary operators for **simple conditional assignments**.
- Leverage ternary in **list comprehensions**, **dictionary comprehensions**, and **lambda functions**.
- Avoid **overusing ternary operators** for complex logic.
- Be mindful of **readability** and **order of operations**.

---

---


The ternary conditional operator, often referred to as the conditional expression, provides a concise way to choose between two expressions based on a condition.

## Syntax

```python
<expression_if_true> if <condition> else <expression_if_false>
```

### Components:
- **expression_if_true**: This expression is evaluated and returned if the condition evaluates to `True`.
- **condition**: A boolean expression that determines which expression to evaluate.
- **expression_if_false**: This expression is evaluated and returned if the condition evaluates to `False`.

## Basic Usage

```python
result = "Even" if x % 2 == 0 else "Odd"
```

- If `x` is even, `result` will be `"Even"`; otherwise, it will be `"Odd"`.

## Ternary in List Comprehensions

You can use the ternary conditional operator within list comprehensions to create lists based on conditions.

```python
# Replace 0 with 'zero' in a list
result = [a if a else 'zero' for a in [0, 1, 0, 3]] # ['zero', 1, 'zero', 3]
```

### Example with More Complex Logic

```python
# Categorizing numbers based on value
numbers = [5, 0, -3, -1, 4]
categories = ["Positive" if n > 0 else "Zero" if n == 0 else "Negative" for n in numbers]
print(categories)  # Output: ['Positive', 'Zero', 'Negative', 'Negative', 'Positive']
```

## Nested Ternary Conditionals

You can nest ternary conditional statements, but readability might suffer.

```python
# Nested ternary for grading
score = 85
grade = "A" if score >= 90 else "B" if score >= 80 else "C"
```

### Example of Nested Ternary

```python
# Assigning letter grades
def assign_grade(score):
    return "A" if score >= 90 else "B" if score >= 80 else "C" if score >= 70 else "F"

print(assign_grade(85))  # Output: 'B'
```

## Use Cases

### 1. Default Value Assignment

```python
x = None
result = x if x is not None else "Default Value"
```

### 2. Function Return Simplification

Using a ternary conditional allows for more compact function return statements:

```python
def check_even_odd(num):
    return "Even" if num % 2 == 0 else "Odd"
```

### 3. Conditional Tuple Selection

```python
result = (x, 100) if x > 50 else (x, 0)  # Select between two tuples based on the condition
```

## Best Practices

1. **Readability**: Avoid excessive nesting of ternary operators. If the expression becomes too complex, consider using traditional `if-else` blocks for clarity.

    ```python
    # Less readable due to nesting
    result = "High" if score > 80 else "Medium" if score > 50 else "Low"
    
    # More readable
    if score > 80:
        result = "High"
    elif score > 50:
        result = "Medium"
    else:
        result = "Low"
    ```

2. **Use with Comprehensions**: The ternary operator is ideal for inline operations, especially within list comprehensions, but ensure it's used in a way that maintains the readability of your code.

3. **Be Cautious with Side Effects**: Avoid placing expressions with side effects in ternary operations. Stick to simple evaluations that don't alter the state of variables if possible.

4. **Documentation**: When using complex conditions, consider documenting your logic in comments to help other developers (or your future self) understand the rationale for the decisions made.

## Conclusion

The ternary conditional operator is a powerful feature in Python that allows for concise expressions based on conditions. However, it is essential to balance conciseness and readability in your code, especially when dealing with more complex conditions or nested ternary operations. Use this cheat sheet to guide your use of ternary conditions in Python programming effectively!

--- 

Map Filter Reduce
------
```python
from functools import reduce
list(map(lambda x: x + 1, range(10)))            # [1, 2, 3, 4, 5, 6, 7, 8, 9,10]
list(filter(lambda x: x > 5, range(10)))         # (6, 7, 8, 9)
reduce(lambda acc, x: acc + x, range(10))        # 45
```

---

### **1. `map()`: Advanced Usage**
#### **Multi-Iterable Mapping**
```python
# Element-wise addition of two lists
nums1 = [1, 2, 3]
nums2 = [4, 5, 6]
result = list(map(lambda x, y: x + y, nums1, nums2))  # [5, 7, 9]
```

#### **Type Conversion**
```python
# Convert strings to integers
str_nums = ['1', '2', '3']
int_nums = list(map(int, str_nums))  # [1, 2, 3]
```

#### **Chaining Operations**
```python
# Square and then convert to string
numbers = [1, 2, 3]
result = list(map(str, map(lambda x: x**2, numbers)))  # ['1', '4', '9']
```

#### **Parallel Processing (with `multiprocessing`)**
```python
from multiprocessing import Pool

def square(x):
    return x ** 2

with Pool(4) as p:
    result = p.map(square, range(10))  # [0, 1, 4, 9, 16, ..., 81]
```

---

### **2. `filter()`: Advanced Usage**
#### **Combining with `map()`**
```python
# Filter even numbers and square them
numbers = [1, 2, 3, 4, 5]
result = list(map(lambda x: x**2, filter(lambda x: x % 2 == 0, numbers)))  # [4, 16]
```

#### **Inverse Filter with `itertools.filterfalse`**
```python
from itertools import filterfalse

numbers = [1, 2, 3, 4, 5]
result = list(filterfalse(lambda x: x > 3, numbers))  # [1, 2, 3]
```

#### **Filtering Complex Objects**
```python
# Filter dictionaries with specific keys
users = [{'name': 'Alice', 'age': 25}, {'name': 'Bob', 'age': 17}]
adults = list(filter(lambda u: u['age'] >= 18, users))  # [{'name': 'Alice', 'age': 25}]
```

---

### **3. `reduce()`: Advanced Usage**
#### **Custom Initial Value**
```python
from functools import reduce

# Sum with initial value
numbers = [1, 2, 3]
sum_result = reduce(lambda acc, x: acc + x, numbers, 10)  # 16 (10 + 1 + 2 + 3)
```

#### **Complex Aggregation**
```python
# Flatten a nested list
nested_list = [[1, 2], [3, 4], [5, 6]]
flattened = reduce(lambda acc, x: acc + x, nested_list, [])  # [1, 2, 3, 4, 5, 6]
```

#### **Dictionary Merging**
```python
# Merge multiple dictionaries
dicts = [{'a': 1}, {'b': 2}, {'a': 3, 'c': 4}]
merged = reduce(lambda d1, d2: {**d1, **d2}, dicts)  # {'a': 3, 'b': 2, 'c': 4}
```

---

### **4. Performance Considerations**
#### **Lazy Evaluation with Iterators**
```python
# Memory-efficient processing (no intermediate lists)
numbers = range(1_000_000)
squared = map(lambda x: x**2, numbers)  # Returns an iterator
```

#### **Using `operator` Module for Speed**
```python
from operator import add, mul
from functools import reduce

# Faster than lambda x, y: x + y
sum_result = reduce(add, range(10))      # 45
product_result = reduce(mul, range(1, 5))  # 24 (1*2*3*4)
```

---

### **5. Real-World Use Cases**
#### **Data Pipeline**
```python
# Read → Clean → Transform → Aggregate
data = ["10", "20", "invalid", "30"]

cleaned = filter(str.isdigit, data)               # ["10", "20", "30"]
transformed = map(int, cleaned)                   # [10, 20, 30]
total = reduce(lambda acc, x: acc + x, transformed)  # 60
```

#### **Functional Composition**
```python
from functools import reduce

# Compose multiple functions
def compose(*funcs):
    return reduce(lambda f, g: lambda x: f(g(x)), funcs)

add_1 = lambda x: x + 1
mul_2 = lambda x: x * 2
func = compose(add_1, mul_2)  # f(x) = add_1(mul_2(x))
print(func(3))                # 7 (3*2 + 1)
```

---

### **6. Best Practices**
1. **Prefer List Comprehensions for Simple Cases**:
   - `[x**2 for x in numbers]` is more readable than `map(lambda x: x**2, numbers)`.

2. **Use Type Hints**:
   ```python
   from typing import Callable, Iterable, Any
   def process_data(func: Callable[[Any], Any], data: Iterable) -> list:
       return list(map(func, data))
   ```

3. **Error Handling**:
   ```python
   def safe_divide(x):
       try:
           return x / 0
       except ZeroDivisionError:
           return float('inf')

   result = list(map(safe_divide, [1, 2, 0]))  # [inf, inf, inf]
   ```

4. **Avoid Overusing `reduce`**:
   - Explicit loops are often more readable for complex operations.

---

### **7. Modern Alternatives**
#### **Generator Expressions**
```python
# Memory-efficient alternative to map/filter
squared_evens = (x**2 for x in range(10) if x % 2 == 0)
```

#### **Pandas/Numpy for Data-Intensive Tasks**
```python
import pandas as pd

# Vectorized operations (faster than map)
df = pd.DataFrame({'values': [1, 2, 3]})
df['squared'] = df['values'].map(lambda x: x**2)
```

---

### **Summary**
- **`map`**: For element-wise transformations (supports multiple iterables).
- **`filter`**: For conditional filtering (use `itertools.filterfalse` for inverse logic).
- **`reduce`**: For aggregation tasks (always start with an initial value for safety).
- Combine with `operator` module and generator expressions for cleaner code.
- Prefer comprehensions for simple cases, but use `map/filter` for lazy evaluation.



#### 1. **`map` Function**
   - Applies a function to all items in an iterable (e.g., list, tuple).
   - Returns a map object (iterator).

   **Basic Example:**
   ```python
   list(map(lambda x: x + 1, range(10)))  # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
   ```

   **Advanced Examples:**
   - **Multiple Iterables:**
     ```python
     list(map(lambda x, y: x + y, [1, 2, 3], [4, 5, 6]))  # [5, 7, 9]
     ```

   - **Mapping with a Custom Function:**
     ```python
     def square(x):
         return x ** 2
     list(map(square, range(5)))  # [0, 1, 4, 9, 16]
     ```

   - **Mapping with Type Conversion:**
     ```python
     list(map(int, ["1", "2", "3"]))  # [1, 2, 3]
     ```

---

#### 2. **`filter` Function**
   - Filters elements from an iterable based on a condition.
   - Returns a filter object (iterator).

   **Basic Example:**
   ```python
   list(filter(lambda x: x > 5, range(10)))  # [6, 7, 8, 9]
   ```

   **Advanced Examples:**
   - **Filtering with a Custom Function:**
     ```python
     def is_even(x):
         return x % 2 == 0
     list(filter(is_even, range(10)))  # [0, 2, 4, 6, 8]
     ```

   - **Filtering Non-Empty Strings:**
     ```python
     list(filter(None, ["", "hello", "", "world"]))  # ["hello", "world"]
     ```

   - **Filtering with Multiple Conditions:**
     ```python
     list(filter(lambda x: x > 2 and x < 8, range(10)))  # [3, 4, 5, 6, 7]
     ```

---

#### 3. **`reduce` Function**
   - Applies a function cumulatively to the items of an iterable, reducing it to a single value.
   - Requires `from functools import reduce`.

   **Basic Example:**
   ```python
   from functools import reduce
   reduce(lambda acc, x: acc + x, range(10))  # 45 (sum of 0 to 9)
   ```

   **Advanced Examples:**
   - **Product of a List:**
     ```python
     reduce(lambda acc, x: acc * x, [1, 2, 3, 4])  # 24 (1 * 2 * 3 * 4)
     ```

   - **Finding the Maximum Value:**
     ```python
     reduce(lambda acc, x: x if x > acc else acc, [10, 20, 5, 30])  # 30
     ```

   - **Concatenating Strings:**
     ```python
     reduce(lambda acc, x: acc + " " + x, ["hello", "world", "!"])  # "hello world !"
     ```

   - **Custom Accumulator Logic:**
     ```python
     reduce(lambda acc, x: acc + [x * 2], [1, 2, 3], [])  # [2, 4, 6] (doubling elements)
     ```

---

#### 4. **Combining `map`, `filter`, and `reduce`**
   - These functions can be combined for more complex operations.

   **Example: Sum of Squares of Even Numbers:**
   ```python
   from functools import reduce
   numbers = range(10)
   even_numbers = filter(lambda x: x % 2 == 0, numbers)  # [0, 2, 4, 6, 8]
   squared_numbers = map(lambda x: x ** 2, even_numbers)  # [0, 4, 16, 36, 64]
   sum_of_squares = reduce(lambda acc, x: acc + x, squared_numbers)  # 120
   ```

   **Example: Flattening a List of Lists:**
   ```python
   from functools import reduce
   lists = [[1, 2], [3, 4], [5, 6]]
   flattened = reduce(lambda acc, x: acc + x, lists)  # [1, 2, 3, 4, 5, 6]
   ```

---

#### 5. **Performance Considerations**
   - **`map` and `filter`:** Return iterators, so they are memory-efficient for large datasets.
   - **`reduce`:** Can be less readable for complex operations; consider using loops or list comprehensions for clarity.

---

#### 6. **Alternatives**
   - **List Comprehensions:** Often more readable than `map` and `filter`.
     ```python
     [x + 1 for x in range(10)]  # Equivalent to map
     [x for x in range(10) if x > 5]  # Equivalent to filter
     ```

   - **Built-in Functions:** Use `sum`, `max`, `min`, etc., instead of `reduce` when possible.
     ```python
     sum(range(10))  # 45 (equivalent to reduce example)
     ```

---



Any All
------
```python
any([False, True, False])# True if at least one item in collection is truthy, False if empty.
all([True,1,3,True])     # True if all items in collection are true
```
----

### **1. Basic Syntax**
```python
any(iterable)  # Returns True if at least one element is truthy
all(iterable)  # Returns True if all elements are truthy (or empty)
```

---

### **2. Truthy/Falsy Evaluation**
- **Falsy values**: `False`, `0`, `""`, `None`, `[]`, `{}`, `()`, etc.
- **Truthy values**: Everything else (including `True`, `1`, `"abc"`, non-empty collections).

---

### **3. Key Behaviors**
| Scenario               | `any()` Result | `all()` Result |
|------------------------|----------------|----------------|
| All elements truthy    | `True`         | `True`         |
| At least one truthy    | `True`         | `False`        |
| All elements falsy     | `False`        | `False`        |
| Empty iterable         | `False`        | `True`         |

---

### **4. Examples**
#### **Basic Usage**
```python
# any()
any([False, 0, ""])          # False
any([False, 1, ""])          # True (1 is truthy)

# all()
all([1, "hello", [1, 2]])    # True
all([1, "", [1, 2]])         # False (empty string is falsy)
```

#### **Edge Cases**
```python
# Empty iterables
any([])                       # False
all([])                       # True

# Nested iterables
any([0, [False], None])       # True ([False] is truthy)
all([1, [0], "a"])           # True (non-empty collections are truthy)
```

---

### **5. With Different Data Types**
#### **Strings**
```python
any(["", " ", "a"])           # True
all(["a", "b", " "])          # True
```

#### **Numbers**
```python
any([0, 0.0, 5])             # True
all([1, 2.5, 3])             # True
```

#### **Dictionaries**
```python
# Checks keys (not values)
any({0: "a", False: "b"})    # False (keys are falsy)
all({1: "a", "b": "c"})      # True (keys are truthy)
```

#### **Sets**
```python
any({0, False, None})        # False
all({1, "a", True})          # True
```

---

### **6. Advanced Use Cases**
#### **Short-Circuit Evaluation**
```python
# Stops at first truthy (any) or falsy (all)
any([0, 1, print("Hi")])     # Prints nothing (stops at 1)
all([1, 0, print("Hi")])     # Prints nothing (stops at 0)
```

#### **With Generators**
```python
# Memory-efficient checks
any(x > 10 for x in range(5))  # False
all(x % 2 == 0 for x in [2, 4, 6])  # True
```

#### **Data Validation**
```python
# Check if all required fields exist in a dictionary
data = {"name": "Alice", "age": 30, "email": "alice@example.com"}
required_fields = ["name", "age", "email"]
all(field in data for field in required_fields)  # True
```

#### **Combining with `map()`**
```python
# Check if any numbers are even
numbers = [1, 3, 5, 8]
any(map(lambda x: x % 2 == 0, numbers))  # True
```

---

### **7. Practical Applications**
#### **Conditional Workflows**
```python
# Only proceed if all files are valid
if all(is_valid(file) for file in files):
    process_files()
```

#### **Search Patterns**
```python
# Check if any substring exists in a text
keywords = ["error", "warning", "critical"]
log = "System encountered a critical error"
any(keyword in log for keyword in keywords)  # True
```

#### **Matrix Operations**
```python
# Check if all rows in a matrix have the same length
matrix = [[1, 2], [3, 4], [5, 6]]
all(len(row) == len(matrix[0]) for row in matrix)  # True
```

---

### **8. Gotchas**
#### **Empty Iterables**
```python
all([])  # True (vacuously true)
any([])  # False
```

#### **Mixed Types**
```python
all([1, "a", [0]])  # True (all are truthy)
any([0, "", None])   # False (all are falsy)
```

#### **Side Effects**
```python
# Side effects in generator expressions
def log(x):
    print(f"Checking {x}")
    return x > 2

any(log(x) for x in [1, 2, 3])  # Prints "Checking 1", "Checking 2", "Checking 3"
```

---

### **9. Performance Tips**
- Use **generator expressions** instead of lists for large datasets (lazy evaluation).
- Place **most likely truthy elements first** in `any()` checks for short-circuiting.
- Place **most likely falsy elements first** in `all()` checks for short-circuiting.

---

### **Summary Table**
| Function | Returns `True` If                     | Empty Iterable | Short-Circuits? |
|----------|---------------------------------------|----------------|------------------|
| `any()`  | At least one element is truthy        | `False`        | Yes (at first truthy) |
| `all()`  | All elements are truthy (or empty)    | `True`         | Yes (at first falsy)  |




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

---

### Advanced Cheatsheet: Closures in Python

#### 1. **What is a Closure?**
   - A **closure** is a nested function that captures and remembers the values of variables from its enclosing scope, even after the outer function has finished executing.
   - Closures are created when:
     1. A nested function references a value from its enclosing function.
     2. The enclosing function returns the nested function.

   **Syntax:**
   ```python
   def outer_function(arg):
       def inner_function():
           # Use arg or other variables from outer_function
           return arg
       return inner_function
   ```

   **Example:**
   ```python
   def get_multiplier(a):
       def out(b):
           return a * b
       return out

   multiply_by_3 = get_multiplier(3)
   print(multiply_by_3(10))  # Output: 30
   ```

---

#### 2. **Key Characteristics of Closures**
   - **Captured Variables:** The nested function retains access to the variables of the enclosing scope, even after the enclosing function has completed execution.
   - **State Encapsulation:** Closures allow you to encapsulate state (data) within a function, making it useful for creating function factories or callbacks.

   **Example:**
   ```python
   def counter():
       count = 0
       def increment():
           nonlocal count
           count += 1
           return count
       return increment

   c = counter()
   print(c(), c(), c())  # Output: 1 2 3
   ```

---

#### 3. **Shared Variables in Closures**
   - If multiple nested functions within the same enclosing function reference the same variable, they share that variable.

   **Example:**
   ```python
   def outer():
       x = 10
       def inner1():
           nonlocal x
           x += 1
           return x
       def inner2():
           nonlocal x
           x *= 2
           return x
       return inner1, inner2

   inc, double = outer()
   print(inc())   # Output: 11
   print(double())  # Output: 22
   print(inc())   # Output: 23
   ```

---

#### 4. **Accessing Closure Variables**
   - You can dynamically access the captured variables of a closure using the `__closure__` attribute.
   - `__closure__` is a tuple of `cell` objects, and each cell has a `cell_contents` attribute that stores the value.

   **Syntax:**
   ```python
   <function>.__closure__[<index>].cell_contents
   ```

   **Example:**
   ```python
   def outer():
       x = 10
       def inner():
           return x
       return inner

   func = outer()
   print(func.__closure__[0].cell_contents)  # Output: 10
   ```

   **Example with Multiple Variables:**
   ```python
   def outer():
       x = 10
       y = 20
       def inner():
           return x, y
       return inner

   func = outer()
   print(func.__closure__[0].cell_contents)  # Output: 10
   print(func.__closure__[1].cell_contents)  # Output: 20
   ```

---

#### 5. **Practical Use Cases for Closures**
   - **Function Factories:** Create specialized functions dynamically.
     ```python
     def power_factory(exponent):
         def power(base):
             return base ** exponent
         return power

     square = power_factory(2)
     cube = power_factory(3)
     print(square(4))  # Output: 16
     print(cube(3))    # Output: 27
     ```

   - **Callbacks and Event Handlers:** Use closures to pass state to callbacks.
     ```python
     def on_button_click(message):
         def callback():
             print(message)
         return callback

     click_handler = on_button_click("Button clicked!")
     click_handler()  # Output: Button clicked!
     ```

   - **Memoization:** Cache results of expensive computations.
     ```python
     def memoize(func):
         cache = {}
         def wrapper(*args):
             if args not in cache:
                 cache[args] = func(*args)
             return cache[args]
         return wrapper

     @memoize
     def fibonacci(n):
         if n <= 1:
             return n
         return fibonacci(n - 1) + fibonacci(n - 2)

     print(fibonacci(10))  # Output: 55
     ```

---

#### 6. **Advanced Topics**
   - **Modifying Captured Variables:**
     Use the `nonlocal` keyword to modify variables in the enclosing scope.
     ```python
     def counter():
         count = 0
         def increment():
             nonlocal count
             count += 1
             return count
         return increment

     c = counter()
     print(c(), c(), c())  # Output: 1 2 3
     ```

   - **Closures with Lambda Functions:**
     Closures can also be created using lambda functions.
     ```python
     def outer(x):
         return lambda y: x + y

     add_5 = outer(5)
     print(add_5(10))  # Output: 15
     ```

   - **Closures in Loops:**
     Be careful when creating closures inside loops, as they may capture the loop variable incorrectly.
     ```python
     functions = []
     for i in range(3):
         def outer(x):
             def inner():
                 return x
             return inner
         functions.append(outer(i))

     print([f() for f in functions])  # Output: [0, 1, 2]
     ```

---

#### 7. **Best Practices**
   - Use closures to encapsulate state and create clean, reusable code.
   - Avoid modifying captured variables unless necessary, as it can make code harder to understand.
   - Use `nonlocal` to modify variables in the enclosing scope explicitly.

---

#### 8. **Common Mistakes**
   - Forgetting to use `nonlocal` when modifying a captured variable.
   - Creating closures in loops without properly capturing the loop variable.
   - Overusing closures, which can lead to hard-to-debug code.

   **Example of a Mistake:**
   ```python
   functions = []
   for i in range(3):
       def inner():
           return i
       functions.append(inner)

   print([f() for f in functions])  # Output: [2, 2, 2] (all functions capture the same i)
   ```

---


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


#### 1. **Variable Scope Basics**
   - **Local Scope:** Variables defined inside a function. They are accessible only within that function.
   - **Enclosing (Nonlocal) Scope:** Variables defined in an enclosing function (for nested functions).
   - **Global Scope:** Variables defined at the top level of a module or script. They are accessible everywhere.
   - **Built-in Scope:** Predefined names provided by Python (e.g., `print`, `len`).

   **Example:**
   ```python
   x = 10  # Global scope

   def outer():
       y = 20  # Enclosing scope
       def inner():
           z = 30  # Local scope
           print(x, y, z)  # Access global, enclosing, and local variables
       inner()

   outer()  # Output: 10 20 30
   ```

---

#### 2. **`global` Keyword**
   - Used to declare that a variable is in the **global scope**.
   - Allows modifying a global variable from within a function.

   **Syntax:**
   ```python
   global variable_name
   ```

   **Example:**
   ```python
   x = 10  # Global variable

   def modify_global():
       global x
       x = 20  # Modify the global variable

   modify_global()
   print(x)  # Output: 20
   ```

   **Common Pitfall:**
   ```python
   x = 10

   def try_to_modify():
       x = 20  # Creates a new local variable, does not modify the global x

   try_to_modify()
   print(x)  # Output: 10 (global x is unchanged)
   ```

---

#### 3. **`nonlocal` Keyword**
   - Used to declare that a variable is in the **enclosing (nonlocal) scope**.
   - Allows modifying a variable in the nearest enclosing function (not global).

   **Syntax:**
   ```python
   nonlocal variable_name
   ```

   **Example:**
   ```python
   def outer():
       x = 10  # Enclosing scope
       def inner():
           nonlocal x
           x = 20  # Modify the enclosing scope variable
       inner()
       print(x)  # Output: 20

   outer()
   ```

   **Common Pitfall:**
   ```python
   def outer():
       x = 10
       def inner():
           x = 20  # Creates a new local variable, does not modify the enclosing x
       inner()
       print(x)  # Output: 10 (enclosing x is unchanged)
   ```

---

#### 4. **Scope Resolution Order (LEGB Rule)**
   Python follows the **LEGB rule** to resolve variable names:
   1. **Local (L):** Inside the current function.
   2. **Enclosing (E):** In enclosing functions (for nested functions).
   3. **Global (G):** At the top level of the module or script.
   4. **Built-in (B):** Predefined names provided by Python.

   **Example:**
   ```python
   x = "global"

   def outer():
       x = "enclosing"
       def inner():
           x = "local"
           print(x)  # Output: local
       inner()
       print(x)  # Output: enclosing

   outer()
   print(x)  # Output: global
   ```

---

#### 5. **Closures and Nonlocal Variables**
   - A **closure** is a nested function that captures and remembers the values of variables from its enclosing scope, even after the outer function has finished executing.
   - Use `nonlocal` to modify variables in the enclosing scope.

   **Example:**
   ```python
   def counter():
       count = 0
       def increment():
           nonlocal count
           count += 1
           return count
       return increment

   c = counter()
   print(c(), c(), c())  # Output: 1 2 3
   ```

---

#### 6. **Advanced Use Cases**
   - **Dynamic Scope Modification:**
     ```python
     def outer():
         x = 10
         def inner():
             nonlocal x
             x += 5
             return x
         return inner

     func = outer()
     print(func())  # Output: 15
     print(func())  # Output: 20
     ```

   - **Global and Nonlocal Together:**
     ```python
     x = "global"

     def outer():
         x = "enclosing"
         def inner():
             global x
             x = "modified global"
         inner()
         print(x)  # Output: enclosing

     outer()
     print(x)  # Output: modified global
     ```

   - **Avoiding Variable Shadowing:**
     ```python
     x = 10

     def outer():
         x = 20
         def inner():
             nonlocal x
             print(x)  # Output: 20
         inner()

     outer()
     print(x)  # Output: 10 (global x is unchanged)
     ```

---

#### 7. **Best Practices**
   - Use `global` sparingly, as it can make code harder to understand and debug.
   - Prefer `nonlocal` over `global` for nested functions.
   - Avoid variable shadowing (using the same variable name in different scopes) to prevent confusion.
   - Use closures to encapsulate state and create clean, reusable code.

---

#### 8. **Common Mistakes**
   - Forgetting to declare a variable as `global` or `nonlocal` when modifying it.
   - Accidentally shadowing a variable by using the same name in a local scope.
   - Assuming a variable is global when it is actually local.

   **Example of a Mistake:**
   ```python
   x = 10

   def modify():
       x = 20  # Creates a new local variable
       print(x)  # Output: 20

   modify()
   print(x)  # Output: 10 (global x is unchanged)
   ```

---



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


---

### Advanced Cheatsheet: Modules in Python

#### 1. **What is a Module?**
   - A **module** is a file containing Python code (e.g., functions, classes, variables).
   - Modules allow you to organize code into reusable components.
   - You can import modules using the `import` statement.

   **Example:**
   ```python
   # my_module.py
   def greet(name):
       return f"Hello, {name}!"

   PI = 3.14159
   ```

   ```python
   # main.py
   import my_module

   print(my_module.greet("Alice"))  # Output: Hello, Alice!
   print(my_module.PI)              # Output: 3.14159
   ```

---

#### 2. **Importing Modules**
   - **Basic Import:**
     ```python
     import module_name
     ```

   - **Import Specific Functions/Classes:**
     ```python
     from module_name import function_name
     ```

   - **Import with Aliasing:**
     ```python
     import module_name as alias
     from module_name import function_name as alias
     ```

   - **Import Everything (Not Recommended):**
     ```python
     from module_name import *
     ```

   **Examples:**
   ```python
   import math
   print(math.sqrt(16))  # Output: 4.0

   from math import sqrt
   print(sqrt(25))       # Output: 5.0

   import math as m
   print(m.pi)           # Output: 3.141592653589793

   from math import pi as PI
   print(PI)             # Output: 3.141592653589793
   ```

---

#### 3. **`if __name__ == '__main__':`**
   - This construct allows you to execute code only when the script is run directly, not when it is imported as a module.
   - Useful for writing reusable modules with test code or main functionality.

   **Example:**
   ```python
   # my_module.py
   def main():
       print("Running main function")

   if __name__ == '__main__':
       main()  # Runs only if the script is executed directly
   ```

   ```bash
   $ python my_module.py
   Running main function
   ```

   ```python
   # main.py
   import my_module  # main() is not executed
   ```

---

#### 4. **Organizing Modules**
   - **Packages:** A collection of modules in a directory with an `__init__.py` file.
   - **Submodules:** Modules within a package.

   **Example:**
   ```
   my_package/
       __init__.py
       module1.py
       module2.py
   ```

   ```python
   # my_package/module1.py
   def func1():
       return "Function 1"

   # my_package/module2.py
   def func2():
       return "Function 2"
   ```

   ```python
   # main.py
   from my_package import module1, module2

   print(module1.func1())  # Output: Function 1
   print(module2.func2())  # Output: Function 2
   ```

---

#### 5. **Advanced Import Techniques**
   - **Relative Imports:** Import modules relative to the current package.
     ```python
     from . import module_name  # Import from the same package
     from .. import module_name # Import from the parent package
     ```

   - **Dynamic Imports:** Import modules dynamically using `importlib`.
     ```python
     import importlib

     module_name = "math"
     module = importlib.import_module(module_name)
     print(module.sqrt(9))  # Output: 3.0
     ```

   - **Lazy Imports:** Delay importing a module until it is actually used.
     ```python
     def lazy_import():
         import math
         return math.sqrt(16)

     print(lazy_import())  # Output: 4.0
     ```

---

#### 6. **Module Attributes**
   - Every module has built-in attributes that provide metadata.
   - Common attributes:
     - `__name__`: The name of the module.
     - `__file__`: The file path of the module.
     - `__doc__`: The module's docstring.

   **Example:**
   ```python
   import math

   print(math.__name__)  # Output: math
   print(math.__file__)  # Output: path/to/math.py (or built-in)
   print(math.__doc__)   # Output: This module provides access to...
   ```

---

#### 7. **Custom Module Initialization**
   - Use the `__init__.py` file to initialize a package or define what gets imported when using `from package import *`.

   **Example:**
   ```python
   # my_package/__init__.py
   from .module1 import func1
   from .module2 import func2

   __all__ = ['func1', 'func2']  # Controls what gets imported with `from package import *`
   ```

   ```python
   # main.py
   from my_package import *

   print(func1())  # Output: Function 1
   print(func2())  # Output: Function 2
   ```

---

#### 8. **Best Practices**
   - **Avoid `from module import *`:** It pollutes the namespace and makes it unclear where functions/classes are coming from.
   - **Use Absolute Imports:** Prefer absolute imports over relative imports for clarity.
   - **Organize Code into Packages:** Group related modules into packages for better maintainability.
   - **Use `if __name__ == '__main__':`:** Ensure test code or main functionality doesn’t run when the module is imported.

---

#### 9. **Common Mistakes**
   - **Circular Imports:** Two modules importing each other, causing an infinite loop.
     ```python
     # module1.py
     from module2 import func2

     def func1():
         return "Function 1"

     # module2.py
     from module1 import func1

     def func2():
         return "Function 2"
     ```

   - **Shadowing Built-in Modules:** Accidentally naming a module the same as a built-in module (e.g., `math.py`).
   - **Missing `__init__.py`:** Forgetting to include `__init__.py` in a package directory.

---

#### 10. **Advanced Use Cases**
   - **Plugin Architecture:** Dynamically load modules to extend functionality.
     ```python
     import importlib

     def load_plugin(plugin_name):
         return importlib.import_module(plugin_name)

     plugin = load_plugin("my_plugin")
     plugin.run()
     ```

   - **Configuration Modules:** Store configuration settings in a module.
     ```python
     # config.py
     DATABASE_URI = "sqlite:///mydb.db"
     DEBUG = True

     # main.py
     from config import DATABASE_URI, DEBUG
     print(DATABASE_URI, DEBUG)
     ```

   - **Monkey Patching:** Modify or extend modules at runtime.
     ```python
     import math

     def new_sqrt(x):
         return x ** 0.5

     math.sqrt = new_sqrt
     print(math.sqrt(16))  # Output: 4.0
     ```

---
Iterators
--------
**In this cheatsheet `'<collection>'` can also mean an iterator.**

```python
<iter> = iter(<collection>)
<iter> = iter(<function>, to_exclusive)     # Sequence of return values until 'to_exclusive'.
<el>   = next(<iter> [, default])           # Raises StopIteration or returns 'default' on end.
```

---

### Iterator Basics Cheatsheet

#### 1. **What is an Iterator?**
   - An **iterator** is an object that allows you to traverse through a collection (e.g., list, tuple, dictionary) or generate values on-the-fly.
   - It implements the **iterator protocol**:
     - `__iter__()`: Returns the iterator object itself.
     - `__next__()`: Returns the next value. Raises `StopIteration` when no more items are available.

---

#### 2. **Creating an Iterator**
   - Use the `iter()` function to create an iterator from a collection.
   - Use the `next()` function to retrieve the next item from the iterator.

   **Syntax:**
   ```python
   <iter> = iter(<collection>)
   <el> = next(<iter> [, default])
   ```

   **Example:**
   ```python
   my_list = [1, 2, 3]
   my_iter = iter(my_list)

   print(next(my_iter))  # Output: 1
   print(next(my_iter))  # Output: 2
   print(next(my_iter))  # Output: 3
   print(next(my_iter, "End"))  # Output: End (default value)
   ```

---

#### 3. **Iterator Protocol**
   - To create a custom iterator, implement the `__iter__()` and `__next__()` methods in a class.

   **Example:**
   ```python
   class CountDown:
       def __init__(self, start):
           self.current = start

       def __iter__(self):
           return self

       def __next__(self):
           if self.current <= 0:
               raise StopIteration
           else:
               self.current -= 1
               return self.current + 1

   countdown = CountDown(3)
   for num in countdown:
       print(num)  # Output: 3 2 1
   ```

---

#### 4. **Using `iter()` with a Function**
   - You can create an iterator from a function using a **sentinel value**. The iterator stops when the function returns the sentinel.

   **Syntax:**
   ```python
   <iter> = iter(<function>, sentinel)
   ```

   **Example:**
   ```python
   import random

   def random_number():
       return random.randint(1, 10)

   random_iter = iter(random_number, 7)  # Stops when random_number() returns 7
   for num in random_iter:
       print(num)  # Output: Random numbers until 7 is generated
   ```

---

#### 5. **Handling `StopIteration`**
   - When an iterator is exhausted, `next()` raises `StopIteration`. You can provide a default value to avoid the error.

   **Example:**
   ```python
   my_iter = iter([1, 2, 3])

   print(next(my_iter))        # Output: 1
   print(next(my_iter))        # Output: 2
   print(next(my_iter))        # Output: 3
   print(next(my_iter, "End"))  # Output: End (default value)
   ```

---

#### 6. **Iterators vs Iterables**
   - **Iterable:** An object that can be looped over (e.g., list, tuple, string). It implements `__iter__()`.
   - **Iterator:** An object that keeps track of the current state during iteration. It implements `__iter__()` and `__next__()`.

   **Example:**
   ```python
   my_list = [1, 2, 3]  # Iterable
   my_iter = iter(my_list)  # Iterator

   for item in my_list:  # Implicitly calls iter(my_list)
       print(item)
   ```

---

#### 7. **Generators as Iterators**
   - **Generators** are a simple way to create iterators using the `yield` keyword.
   - They automatically implement the iterator protocol.

   **Example:**
   ```python
   def countdown(start):
       while start > 0:
           yield start
           start -= 1

   for num in countdown(3):
       print(num)  # Output: 3 2 1
   ```

---

#### 8. **Common Iterator Use Cases**
   - **Lazy Evaluation:** Process data on-the-fly without loading everything into memory.
   - **Infinite Sequences:** Generate an infinite sequence of values.
   - **Pipeline Processing:** Chain multiple iterators together for efficient data processing.

   **Example: Lazy Evaluation**
   ```python
   def read_large_file(file_path):
       with open(file_path, "r") as file:
           for line in file:
               yield line.strip()

   for line in read_large_file("large_file.txt"):
       print(line)  # Processes one line at a time
   ```

   **Example: Infinite Sequence**
   ```python
   import itertools

   counter = itertools.count(start=1, step=2)  # Infinite iterator: 1, 3, 5, 7, ...
   for _ in range(5):
       print(next(counter))  # Output: 1 3 5 7 9
   ```

---

#### 9. **Key Functions from `itertools`**
   - The `itertools` module provides powerful tools for working with iterators.

   **Common Functions:**
   - `itertools.count()`: Infinite sequence of numbers.
   - `itertools.cycle()`: Cycle through a sequence infinitely.
   - `itertools.repeat()`: Repeat an element infinitely or a specified number of times.
   - `itertools.chain()`: Chain multiple iterables together.
   - `itertools.islice()`: Slice an iterator.

   **Examples:**
   ```python
   import itertools

   # Example 1: count
   counter = itertools.count(start=10, step=-1)
   print([next(counter) for _ in range(5)])  # Output: [10, 9, 8, 7, 6]

   # Example 2: cycle
   cycler = itertools.cycle("ABC")
   print([next(cycler) for _ in range(5)])  # Output: ['A', 'B', 'C', 'A', 'B']

   # Example 3: chain
   chained = itertools.chain([1, 2], [3, 4])
   print(list(chained))  # Output: [1, 2, 3, 4]

   # Example 4: islice
   sliced = itertools.islice(itertools.count(), 5)  # First 5 numbers
   print(list(sliced))  # Output: [0, 1, 2, 3, 4]
   ```

---

#### 10. **Best Practices**
   - Use iterators for large datasets to save memory.
   - Prefer generators for creating custom iterators.
   - Use `itertools` for efficient iterator operations.
   - Handle `StopIteration` gracefully or use a `default` value with `next()`.

---

#### 11. **Common Mistakes**
   - **Exhausting an Iterator:** Iterators can only be traversed once.
     ```python
     my_iter = iter([1, 2, 3])
     print(list(my_iter))  # Output: [1, 2, 3]
     print(list(my_iter))  # Output: [] (iterator is exhausted)
     ```

   - **Forgetting to Implement `__iter__()`:** Custom iterators must implement both `__iter__()` and `__next__()`.

---


---

### Advanced Cheatsheet: Iterators in Python

#### 1. **What is an Iterator?**
   - An **iterator** is an object that implements the **iterator protocol**:
     - `__iter__()`: Returns the iterator object itself.
     - `__next__()`: Returns the next value from the iterator. Raises `StopIteration` when there are no more items.
   - Iterators are used to traverse collections (e.g., lists, tuples, dictionaries) or generate sequences on-the-fly.

   **Example:**
   ```python
   my_list = [1, 2, 3]
   my_iter = iter(my_list)

   print(next(my_iter))  # Output: 1
   print(next(my_iter))  # Output: 2
   print(next(my_iter))  # Output: 3
   print(next(my_iter, "End"))  # Output: End (default value)
   ```

---

#### 2. **Creating Iterators**
   - **Using `iter()`:**
     - Convert a collection (e.g., list, tuple, string) into an iterator.
     - Create an iterator from a function using the `sentinel` value.

   **Syntax:**
   ```python
   <iter> = iter(<collection>)
   <iter> = iter(<function>, sentinel)  # Iterator stops when function returns sentinel
   ```

   **Examples:**
   ```python
   # Example 1: Basic iterator
   my_list = [1, 2, 3]
   my_iter = iter(my_list)
   print(next(my_iter))  # Output: 1

   # Example 2: Iterator with sentinel
   import random
   def random_number():
       return random.randint(1, 10)

   random_iter = iter(random_number, 7)  # Stops when random_number() returns 7
   for num in random_iter:
       print(num)  # Output: Random numbers until 7 is generated
   ```

---

#### 3. **Using `next()`**
   - Retrieve the next item from an iterator.
   - If the iterator is exhausted, it raises `StopIteration` unless a `default` value is provided.

   **Syntax:**
   ```python
   <el> = next(<iter> [, default])
   ```

   **Examples:**
   ```python
   my_iter = iter([1, 2, 3])

   print(next(my_iter))        # Output: 1
   print(next(my_iter))        # Output: 2
   print(next(my_iter))        # Output: 3
   print(next(my_iter, "End"))  # Output: End (default value)
   ```

---

#### 4. **Custom Iterators**
   - Create your own iterator by implementing the `__iter__()` and `__next__()` methods.

   **Example:**
   ```python
   class CountDown:
       def __init__(self, start):
           self.current = start

       def __iter__(self):
           return self

       def __next__(self):
           if self.current <= 0:
               raise StopIteration
           else:
               self.current -= 1
               return self.current + 1

   countdown = CountDown(3)
   for num in countdown:
       print(num)  # Output: 3 2 1
   ```

---

#### 5. **Infinite Iterators**
   - Iterators that never exhaust (e.g., `itertools.count()`).

   **Example:**
   ```python
   import itertools

   counter = itertools.count(start=1, step=2)  # Infinite iterator: 1, 3, 5, 7, ...
   for _ in range(5):
       print(next(counter))  # Output: 1 3 5 7 9
   ```

---

#### 6. **Itertools Module**
   - The `itertools` module provides powerful tools for working with iterators.

   **Common Functions:**
   - `itertools.count()`: Infinite sequence of numbers.
   - `itertools.cycle()`: Cycle through a sequence infinitely.
   - `itertools.repeat()`: Repeat an element infinitely or a specified number of times.
   - `itertools.chain()`: Chain multiple iterables together.
   - `itertools.islice()`: Slice an iterator.

   **Examples:**
   ```python
   import itertools

   # Example 1: count
   counter = itertools.count(start=10, step=-1)
   print([next(counter) for _ in range(5)])  # Output: [10, 9, 8, 7, 6]

   # Example 2: cycle
   cycler = itertools.cycle("ABC")
   print([next(cycler) for _ in range(5)])  # Output: ['A', 'B', 'C', 'A', 'B']

   # Example 3: chain
   chained = itertools.chain([1, 2], [3, 4])
   print(list(chained))  # Output: [1, 2, 3, 4]

   # Example 4: islice
   sliced = itertools.islice(itertools.count(), 5)  # First 5 numbers
   print(list(sliced))  # Output: [0, 1, 2, 3, 4]
   ```

---

#### 7. **Generators as Iterators**
   - **Generators** are a concise way to create iterators using the `yield` keyword.
   - They automatically implement the iterator protocol.

   **Example:**
   ```python
   def countdown(start):
       while start > 0:
           yield start
           start -= 1

   for num in countdown(3):
       print(num)  # Output: 3 2 1
   ```

---

#### 8. **Advanced Use Cases**
   - **Lazy Evaluation:** Iterators allow you to process data on-the-fly without loading everything into memory.
   - **Pipeline Processing:** Chain multiple iterators together for efficient data processing.

   **Example: Lazy Evaluation**
   ```python
   def read_large_file(file_path):
       with open(file_path, "r") as file:
           for line in file:
               yield line.strip()

   for line in read_large_file("large_file.txt"):
       print(line)  # Processes one line at a time
   ```

   **Example: Pipeline Processing**
   ```python
   def squares(numbers):
       for num in numbers:
           yield num ** 2

   def evens(numbers):
       for num in numbers:
           if num % 2 == 0:
               yield num

   numbers = range(10)
   pipeline = evens(squares(numbers))
   print(list(pipeline))  # Output: [0, 4, 16, 36, 64]
   ```

---

#### 9. **Best Practices**
   - Use iterators for large datasets to save memory.
   - Prefer generators for creating custom iterators.
   - Use `itertools` for efficient iterator operations.
   - Handle `StopIteration` gracefully or use a `default` value with `next()`.

---

#### 10. **Common Mistakes**
   - **Exhausting an Iterator:** Iterators can only be traversed once.
     ```python
     my_iter = iter([1, 2, 3])
     print(list(my_iter))  # Output: [1, 2, 3]
     print(list(my_iter))  # Output: [] (iterator is exhausted)
     ```

   - **Forgetting to Implement `__iter__()`:** Custom iterators must implement both `__iter__()` and `__next__()`.

---

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
---

#### 1. **What is a Generator?**
   - A **generator** is a special type of iterator that allows you to iterate over a sequence of values without storing them in memory.
   - Generators are created using functions with the `yield` keyword.
   - They automatically implement the **iterator protocol** (`__iter__()` and `__next__()`).

   **Example:**
   ```python
   def simple_generator():
       yield 1
       yield 2
       yield 3

   gen = simple_generator()
   print(next(gen))  # Output: 1
   print(next(gen))  # Output: 2
   print(next(gen))  # Output: 3
   ```

---

#### 2. **Basic Generator Syntax**
   - Use the `yield` keyword to produce a value and pause the function's execution.
   - The function resumes execution from where it left off when `next()` is called again.

   **Syntax:**
   ```python
   def generator_function():
       yield <value>
   ```

   **Example:**
   ```python
   def count_up_to(max):
       count = 1
       while count <= max:
           yield count
           count += 1

   counter = count_up_to(3)
   for num in counter:
       print(num)  # Output: 1 2 3
   ```

---

#### 3. **Infinite Generators**
   - Generators can produce an infinite sequence of values.

   **Example:**
   ```python
   def infinite_counter():
       count = 0
       while True:
           yield count
           count += 1

   counter = infinite_counter()
   print(next(counter))  # Output: 0
   print(next(counter))  # Output: 1
   print(next(counter))  # Output: 2
   ```

---

#### 4. **Generator Expressions**
   - A concise way to create generators using a syntax similar to list comprehensions.
   - Use parentheses `()` instead of square brackets `[]`.

   **Syntax:**
   ```python
   gen = (expression for item in iterable)
   ```

   **Example:**
   ```python
   squares = (x ** 2 for x in range(5))
   for num in squares:
       print(num)  # Output: 0 1 4 9 16
   ```

---

#### 5. **Sending Values to Generators**
   - Use the `send()` method to pass values back into a generator.
   - The generator must be designed to receive values using `yield`.

   **Example:**
   ```python
   def accumulator():
       total = 0
       while True:
           value = yield total
           if value is None:
               break
           total += value

   acc = accumulator()
   next(acc)  # Start the generator
   print(acc.send(10))  # Output: 10
   print(acc.send(20))  # Output: 30
   print(acc.send(5))   # Output: 35
   ```

---

#### 6. **Throwing Exceptions into Generators**
   - Use the `throw()` method to raise an exception inside a generator.

   **Example:**
   ```python
   def generator():
       try:
           while True:
               yield "Running"
       except ValueError:
           yield "ValueError caught"

   gen = generator()
   print(next(gen))  # Output: Running
   print(gen.throw(ValueError))  # Output: ValueError caught
   ```

---

#### 7. **Closing Generators**
   - Use the `close()` method to stop a generator prematurely.
   - This raises a `GeneratorExit` exception inside the generator.

   **Example:**
   ```python
   def generator():
       try:
           while True:
               yield "Running"
       except GeneratorExit:
           print("Generator closed")

   gen = generator()
   print(next(gen))  # Output: Running
   gen.close()       # Output: Generator closed
   ```

---

#### 8. **Chaining Generators**
   - Combine multiple generators into a single generator using `yield from`.

   **Syntax:**
   ```python
   yield from <generator>
   ```

   **Example:**
   ```python
   def chain_generators(*generators):
       for gen in generators:
           yield from gen

   gen1 = (x for x in range(3))
   gen2 = (x for x in range(3, 6))
   combined = chain_generators(gen1, gen2)

   for num in combined:
       print(num)  # Output: 0 1 2 3 4 5
   ```

---

#### 9. **Stateful Generators**
   - Generators can maintain state between `yield` calls.

   **Example:**
   ```python
   def fibonacci():
       a, b = 0, 1
       while True:
           yield a
           a, b = b, a + b

   fib = fibonacci()
   for _ in range(10):
       print(next(fib))  # Output: 0 1 1 2 3 5 8 13 21 34
   ```

---

#### 10. **Advanced Use Cases**
   - **Pipeline Processing:** Chain generators together for efficient data processing.
   - **Lazy Evaluation:** Process data on-the-fly without loading everything into memory.
   - **Coroutines:** Use generators to implement coroutines for cooperative multitasking.

   **Example: Pipeline Processing**
   ```python
   def integers():
       for i in range(1, 5):
           yield i

   def squares(seq):
       for num in seq:
           yield num ** 2

   def negated(seq):
       for num in seq:
           yield -num

   pipeline = negated(squares(integers()))
   for num in pipeline:
       print(num)  # Output: -1 -4 -9 -16
   ```

   **Example: Lazy Evaluation**
   ```python
   def read_large_file(file_path):
       with open(file_path, "r") as file:
           for line in file:
               yield line.strip()

   for line in read_large_file("large_file.txt"):
       print(line)  # Processes one line at a time
   ```

---

#### 11. **Best Practices**
   - Use generators for large datasets to save memory.
   - Prefer generator expressions for simple transformations.
   - Use `yield from` to chain generators for cleaner code.
   - Handle exceptions and cleanup using `try`/`except` and `finally`.

---

#### 12. **Common Mistakes**
   - **Exhausting a Generator:** Generators can only be traversed once.
     ```python
     gen = (x for x in range(3))
     print(list(gen))  # Output: [0, 1, 2]
     print(list(gen))  # Output: [] (generator is exhausted)
     ```

   - **Forgetting to Start the Generator:** Use `next()` to start a generator before sending values.
     ```python
     def generator():
         value = yield
         print(value)

     gen = generator()
     gen.send("Hello")  # Error: Can't send non-None value to a just-started generator
     ```

---


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

### Decorators Cheatsheet: From Basic to Professional

#### 1. **What is a Decorator?**
   - A **decorator** is a function that takes another function as input, adds some functionality, and returns the modified function.
   - Decorators are used to extend or modify the behavior of functions or methods without changing their actual code.
   - They are applied using the `@decorator_name` syntax.

   **Example:**
   ```python
   def my_decorator(func):
       def wrapper():
           print("Something before the function.")
           func()
           print("Something after the function.")
       return wrapper

   @my_decorator
   def say_hello():
       print("Hello!")

   say_hello()
   # Output:
   # Something before the function.
   # Hello!
   # Something after the function.
   ```

---

#### 2. **Basic Decorator Syntax**
   - A decorator is a function that takes a function as an argument and returns a new function.
   - The new function (often called a **wrapper**) typically calls the original function and adds additional behavior.

   **Syntax:**
   ```python
   def decorator_name(func):
       def wrapper(*args, **kwargs):
           # Do something before
           result = func(*args, **kwargs)
           # Do something after
           return result
       return wrapper

   @decorator_name
   def function_name():
       ...
   ```

   **Example:**
   ```python
   def uppercase_decorator(func):
       def wrapper(*args, **kwargs):
           result = func(*args, **kwargs)
           return result.upper()
       return wrapper

   @uppercase_decorator
   def greet(name):
       return f"Hello, {name}"

   print(greet("Alice"))  # Output: HELLO, ALICE
   ```

---

#### 3. **Decorators with Arguments**
   - Decorators can accept arguments by adding an outer function.

   **Syntax:**
   ```python
   def decorator_with_args(arg1, arg2):
       def decorator(func):
           def wrapper(*args, **kwargs):
               # Use arg1, arg2
               result = func(*args, **kwargs)
               return result
           return wrapper
       return decorator

   @decorator_with_args(arg1, arg2)
   def function_name():
       ...
   ```

   **Example:**
   ```python
   def repeat(num_times):
       def decorator(func):
           def wrapper(*args, **kwargs):
               for _ in range(num_times):
                   result = func(*args, **kwargs)
               return result
           return wrapper
       return decorator

   @repeat(3)
   def say_hello():
       print("Hello!")

   say_hello()
   # Output:
   # Hello!
   # Hello!
   # Hello!
   ```

---

#### 4. **Chaining Decorators**
   - Multiple decorators can be applied to a single function. They are executed in the order they are listed.

   **Example:**
   ```python
   def decorator1(func):
       def wrapper():
           print("Decorator 1")
           func()
       return wrapper

   def decorator2(func):
       def wrapper():
           print("Decorator 2")
           func()
       return wrapper

   @decorator1
   @decorator2
   def say_hello():
       print("Hello!")

   say_hello()
   # Output:
   # Decorator 1
   # Decorator 2
   # Hello!
   ```

---

#### 5. **Using `functools.wraps`**
   - The `functools.wraps` decorator preserves the original function's metadata (e.g., name, docstring) when using decorators.

   **Syntax:**
   ```python
   from functools import wraps

   def decorator_name(func):
       @wraps(func)
       def wrapper(*args, **kwargs):
           ...
       return wrapper
   ```

   **Example:**
   ```python
   from functools import wraps

   def my_decorator(func):
       @wraps(func)
       def wrapper(*args, **kwargs):
           print("Something before the function.")
           result = func(*args, **kwargs)
           print("Something after the function.")
           return result
       return wrapper

   @my_decorator
   def say_hello():
       """Says hello."""
       print("Hello!")

   print(say_hello.__name__)  # Output: say_hello
   print(say_hello.__doc__)   # Output: Says hello.
   ```

---

#### 6. **Class-Based Decorators**
   - Decorators can also be implemented as classes by defining the `__call__` method.

   **Syntax:**
   ```python
   class DecoratorClass:
       def __init__(self, func):
           self.func = func

       def __call__(self, *args, **kwargs):
           # Do something before
           result = self.func(*args, **kwargs)
           # Do something after
           return result

   @DecoratorClass
   def function_name():
       ...
   ```

   **Example:**
   ```python
   class Timer:
       def __init__(self, func):
           self.func = func

       def __call__(self, *args, **kwargs):
           import time
           start = time.time()
           result = self.func(*args, **kwargs)
           end = time.time()
           print(f"Execution time: {end - start} seconds")
           return result

   @Timer
   def long_running_function():
       import time
       time.sleep(2)

   long_running_function()  # Output: Execution time: 2.00 seconds
   ```

---

#### 7. **Decorators with Optional Arguments**
   - Decorators can be designed to accept optional arguments by checking the type of the first argument.

   **Example:**
   ```python
   def flexible_decorator(arg):
       if callable(arg):  # If no arguments are passed
           func = arg
           def wrapper(*args, **kwargs):
               print("Decorator without arguments")
               return func(*args, **kwargs)
           return wrapper
       else:  # If arguments are passed
           def decorator(func):
               def wrapper(*args, **kwargs):
                   print(f"Decorator with argument: {arg}")
                   return func(*args, **kwargs)
               return wrapper
           return decorator

   @flexible_decorator
   def function1():
       print("Function 1")

   @flexible_decorator("custom_arg")
   def function2():
       print("Function 2")

   function1()  # Output: Decorator without arguments \n Function 1
   function2()  # Output: Decorator with argument: custom_arg \n Function 2
   ```

---

#### 8. **Advanced Use Cases**
   - **Caching/Memoization:** Store the results of expensive function calls.
   - **Logging:** Log function calls and their arguments.
   - **Access Control:** Restrict access to certain functions.
   - **Validation:** Validate function arguments.

   **Example: Caching**
   ```python
   from functools import wraps

   def cache(func):
       cached_results = {}
       @wraps(func)
       def wrapper(*args):
           if args in cached_results:
               return cached_results[args]
           result = func(*args)
           cached_results[args] = result
           return result
       return wrapper

   @cache
   def fibonacci(n):
       if n <= 1:
           return n
       return fibonacci(n - 1) + fibonacci(n - 2)

   print(fibonacci(10))  # Output: 55
   ```

---

#### 9. **Best Practices**
   - Use `functools.wraps` to preserve function metadata.
   - Keep decorators simple and focused on a single responsibility.
   - Use decorators for cross-cutting concerns (e.g., logging, timing, caching).
   - Avoid overusing decorators, as they can make code harder to debug.

---

#### 10. **Common Mistakes**
   - **Forgetting to Return the Result:** Always return the result of the wrapped function.
   - **Overwriting Function Metadata:** Use `functools.wraps` to avoid losing the original function's metadata.
   - **Incorrect Argument Handling:** Ensure the wrapper function accepts `*args` and `**kwargs`.

---
Certainly! Let’s dive into **debugging** in Python, covering everything from basic techniques to advanced tools and practices. This cheatsheet will help you master debugging and write more robust code.

---

### Debugging-----

#### 1. **Basic Debugging Techniques**
   - **Print Debugging:** Use `print()` statements to inspect variables and control flow.
   - **Assertions:** Use `assert` statements to check for conditions that must be true.

   **Example: Print Debugging**
   ```python
   def add(x, y):
       print(f"x: {x}, y: {y}")  # Debugging print
       return x + y

   result = add(3, 5)
   print(result)  # Output: x: 3, y: 5 \n 8
   ```

   **Example: Assertions**
   ```python
   def divide(x, y):
       assert y != 0, "y cannot be zero"
       return x / y

   print(divide(10, 2))  # Output: 5.0
   print(divide(10, 0))  # Raises AssertionError: y cannot be zero
   ```

---

#### 2. **Using `pdb` (Python Debugger)**
   - The built-in `pdb` module allows you to interactively debug your code.
   - Insert `import pdb; pdb.set_trace()` to start debugging at a specific point.

   **Example:**
   ```python
   def add(x, y):
       import pdb; pdb.set_trace()  # Start debugging here
       return x + y

   result = add(3, 5)
   print(result)
   ```

   **Common `pdb` Commands:**
   - `n` (next): Execute the next line.
   - `c` (continue): Continue execution until the next breakpoint.
   - `q` (quit): Quit the debugger.
   - `p <variable>`: Print the value of a variable.
   - `l` (list): Show the current code context.

---

#### 3. **Advanced Debugging Tools**
   - **`logging` Module:** Use the `logging` module for more flexible and configurable debugging output.
   - **IDE Debuggers:** Use debuggers in IDEs like PyCharm, VSCode, or Jupyter Notebook.

   **Example: `logging` Module**
   ```python
   import logging

   logging.basicConfig(level=logging.DEBUG)  # Set logging level

   def add(x, y):
       logging.debug(f"x: {x}, y: {y}")
       return x + y

   result = add(3, 5)
   logging.info(f"Result: {result}")
   # Output:
   # DEBUG:root:x: 3, y: 5
   # INFO:root:Result: 8
   ```

---

#### 4. **Debugging with Decorators**
   - Decorators can be used to add debugging functionality to functions.

   **Example: Debug Decorator**
   ```python
   from functools import wraps

   def debug(func):
       @wraps(func)
       def wrapper(*args, **kwargs):
           print(f"Calling {func.__name__} with args: {args}, kwargs: {kwargs}")
           result = func(*args, **kwargs)
           print(f"{func.__name__} returned: {result}")
           return result
       return wrapper

   @debug
   def add(x, y):
       return x + y

   add(3, 5)
   # Output:
   # Calling add with args: (3, 5), kwargs: {}
   # add returned: 8
   ```

---

#### 5. **Handling Exceptions**
   - Use `try`/`except` blocks to catch and handle exceptions.
   - Use `traceback` to print detailed error information.

   **Example:**
   ```python
   import traceback

   def divide(x, y):
       try:
           return x / y
       except ZeroDivisionError as e:
           print(f"Error: {e}")
           traceback.print_exc()  # Print detailed traceback
           return None

   divide(10, 0)
   # Output:
   # Error: division by zero
   # Traceback (most recent call last):
   #   File "<stdin>", line 4, in divide
   # ZeroDivisionError: division by zero
   ```

---

#### 6. **Profiling and Performance Debugging**
   - Use `cProfile` or `timeit` to measure the performance of your code.

   **Example: `cProfile`**
   ```python
   import cProfile

   def slow_function():
       total = 0
       for i in range(1000000):
           total += i
       return total

   cProfile.run('slow_function()')
   ```

   **Example: `timeit`**
   ```python
   import timeit

   def fast_function():
       return sum(range(1000000))

   print(timeit.timeit(fast_function, number=100))
   ```

---

#### 7. **Advanced Debugging Techniques**
   - **Post-Mortem Debugging:** Automatically start a debugger when an exception occurs.
   - **Remote Debugging:** Debug code running on a remote machine.

   **Example: Post-Mortem Debugging**
   ```python
   def faulty_function():
       return 1 / 0

   import pdb

   try:
       faulty_function()
   except Exception as e:
       pdb.post_mortem()  # Start debugger after exception
   ```

---

#### 8. **Best Practices**
   - **Write Testable Code:** Break your code into small, testable functions.
   - **Use Logging:** Replace `print()` statements with logging for better control.
   - **Handle Exceptions Gracefully:** Use `try`/`except` to catch and handle errors.
   - **Profile Your Code:** Identify performance bottlenecks using profiling tools.

---

#### 9. **Common Mistakes**
   - **Overusing `print()`:** Use logging or a debugger instead of littering your code with `print()` statements.
   - **Ignoring Exceptions:** Always handle exceptions to avoid unexpected crashes.
   - **Not Testing Edge Cases:** Test your code with edge cases to catch hidden bugs.

---


Debugging is a systematic process of finding and fixing issues in code. Depending on the complexity of the issue and your experience, debugging steps can vary. Here's a step-by-step guide that progresses from basic to advanced debugging strategies:

---

### **Basic Debugging Steps:**

1. **Reproduce the Issue:**
   - Before attempting to fix the bug, ensure you can consistently reproduce the issue. This helps you understand the problem and test potential fixes.
   - Example: If you encounter a crash or incorrect output, try to run the code in the same environment and with the same inputs.

2. **Read Error Messages:**
   - Carefully read any error or exception messages. Many times, error messages give you a direct hint about the source of the problem, such as the line number or the type of error (e.g., `IndexError`, `ValueError`, etc.).
   - **Tip**: Check the stack trace. The error’s location is often shown with additional context.

3. **Check Syntax:**
   - Errors in syntax (e.g., missing parentheses, brackets, typos) can lead to problems in your code. Double-check function names, variable names, and punctuation marks like commas, semicolons, or colons.

4. **Print Debugging (or Logging):**
   - Add `print()` statements or logging calls in your code at strategic points to trace the flow of execution. Output the values of variables to understand what’s happening at runtime.
   - **Example**:
     ```python
     print("Variable x:", x)
     ```

5. **Use a Simple Test Case:**
   - Simplify your code and run a minimal test case that isolates the problem. This can help you narrow down whether the issue is with your logic, input, or environment.
   
---

### **Intermediate Debugging Steps:**

6. **Check for Edge Cases:**
   - Consider edge cases or unusual inputs that may be triggering the bug. For example, check if the code works with empty lists, large inputs, or special characters.

7. **Use a Debugger (e.g., `pdb` for Python):**
   - A debugger allows you to step through your code one line at a time. You can inspect variables, set breakpoints, and follow the execution path.
   - **Example** in Python:
     ```python
     import pdb; pdb.set_trace()
     ```
   - This halts the execution at that point, allowing you to manually inspect the state and step through the code.

8. **Check for Off-by-One Errors:**
   - These types of errors often occur when working with loops, arrays, or indexes. Ensure your loops are starting and ending at the correct indices.
   
9. **Check for Logical Errors:**
   - Make sure that your logic is correct. Sometimes the code runs without crashing but produces incorrect results due to a flaw in your algorithm.

10. **Test in Isolation:**
    - If you're dealing with a complex system, test individual parts of your code in isolation (unit testing). This allows you to determine if the bug is in a particular function or module.
    - **Unit Testing Frameworks**: Use tools like `unittest` in Python or `JUnit` in Java to automate tests.

---

### **Advanced Debugging Steps:**

11. **Version Control (Git):**
    - Use version control tools like **Git** to identify changes that may have introduced bugs. Tools like `git bisect` can help you pinpoint exactly when the issue was introduced.
    - **Example**: Run `git bisect` to find the commit that caused the bug.

12. **Use Advanced Debuggers (IDE Debuggers):**
    - Many modern IDEs, like **PyCharm**, **VS Code**, or **Eclipse**, offer advanced debugging tools. These include features like:
      - **Breakpoints**: Pausing execution at specific points.
      - **Watch Variables**: Tracking variable values as you step through the code.
      - **Call Stack**: Inspecting the sequence of function calls leading to an issue.

13. **Check Memory Leaks (if applicable):**
    - In languages like C/C++, memory management is crucial. Use tools like **Valgrind** to detect memory leaks or out-of-bounds errors.

14. **Profiling and Performance Analysis:**
    - If the issue is related to performance, use profiling tools to identify bottlenecks (e.g., **cProfile** in Python, **gprof** for C/C++).
    - Analyze the runtime complexity and optimize slow code sections.

15. **Concurrency and Threading Issues:**
    - If working with multi-threaded or distributed systems, concurrency bugs (e.g., race conditions, deadlocks) can be tricky. Use tools like **thread sanitizers** or logging to track thread behavior.
    - **Advanced Step**: Try running your code with different thread priorities or scheduling methods to catch concurrency issues.

16. **Check for External Dependencies:**
    - In modern software, external libraries or APIs often introduce issues. Ensure the version of libraries you are using is compatible with your code, or verify that external services (e.g., databases, web APIs) are functioning correctly.

17. **Run in a Clean Environment:**
    - Sometimes, bugs can be due to environmental issues (e.g., incorrect libraries, incompatible system configurations). Set up a clean virtual environment or container (e.g., using **Docker** or **virtualenv** for Python) to isolate the problem.

18. **Revisit Assumptions and Documentation:**
    - Review the documentation for the libraries you're using, and confirm that your assumptions about how they work are correct. In complex systems, misreading documentation is a common source of errors.
    
19. **Ask for Help (Stack Overflow, Forums, Peer Review):**
    - If you’ve tried everything and are still stuck, consider reaching out for help. Sometimes, another set of eyes can spot the problem quicker. Share your code and describe the issue on forums like **Stack Overflow** or ask a colleague to review your code.

20. **Revisit Design and Refactor:**
    - If debugging reveals that the code is too complex, inefficient, or prone to errors, it may be time to rethink the design. Refactor the code to make it more maintainable and easier to debug in the future.

---

### **Bonus Tips for Debugging:**

- **Don’t rush the process**: Debugging can be a time-consuming process. Take breaks if needed and approach the issue with a clear mind.
- **Document your findings**: As you debug, document what you’ve tried and the results. This helps you keep track of your steps and makes it easier to explain the problem to others.
- **Learn from bugs**: Each bug you encounter is an opportunity to improve your problem-solving skills. Take the time to understand why a particular issue happened to avoid similar problems in the future.

---

By following these steps from basic to advanced, you'll gradually develop a deeper understanding of debugging techniques and become more efficient at identifying and solving bugs.

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

## Exception Handling



---

### Exception Handling Cheatsheet: From Basic to Advanced

#### 1. **Basic Exception Handling**
   - Use `try`/`except` blocks to catch and handle exceptions.
   - The `try` block contains code that might raise an exception.
   - The `except` block handles the exception if it occurs.

   **Syntax:**
   ```python
   try:
       # Code that might raise an exception
   except <ExceptionType>:
       # Code to handle the exception
   ```

   **Example:**
   ```python
   try:
       result = 10 / 0
   except ZeroDivisionError:
       print("Cannot divide by zero!")
   # Output: Cannot divide by zero!
   ```

---

#### 2. **Handling Multiple Exceptions**
   - Use multiple `except` blocks to handle different types of exceptions.

   **Syntax:**
   ```python
   try:
       # Code that might raise an exception
   except <ExceptionType1>:
       # Handle ExceptionType1
   except <ExceptionType2>:
       # Handle ExceptionType2
   ```

   **Example:**
   ```python
   try:
       num = int("abc")
   except ValueError:
       print("Invalid number!")
   except TypeError:
       print("Type error occurred!")
   # Output: Invalid number!
   ```

---

#### 3. **Catching All Exceptions**
   - Use a bare `except` block to catch all exceptions (not recommended unless necessary).

   **Syntax:**
   ```python
   try:
       # Code that might raise an exception
   except:
       # Handle all exceptions
   ```

   **Example:**
   ```python
   try:
       result = 10 / 0
   except:
       print("An error occurred!")
   # Output: An error occurred!
   ```

---

#### 4. **Accessing Exception Details**
   - Use `as` to access the exception object and its details (e.g., error message).

   **Syntax:**
   ```python
   try:
       # Code that might raise an exception
   except <ExceptionType> as e:
       # Access exception details
   ```

   **Example:**
   ```python
   try:
       result = 10 / 0
   except ZeroDivisionError as e:
       print(f"Error: {e}")
   # Output: Error: division by zero
   ```

---

#### 5. **The `else` Block**
   - The `else` block runs if no exception occurs in the `try` block.

   **Syntax:**
   ```python
   try:
       # Code that might raise an exception
   except <ExceptionType>:
       # Handle the exception
   else:
       # Code to run if no exception occurs
   ```

   **Example:**
   ```python
   try:
       result = 10 / 2
   except ZeroDivisionError:
       print("Cannot divide by zero!")
   else:
       print(f"Result: {result}")
   # Output: Result: 5.0
   ```

---

#### 6. **The `finally` Block**
   - The `finally` block runs **always**, whether an exception occurs or not.
   - Useful for cleanup operations (e.g., closing files, releasing resources).

   **Syntax:**
   ```python
   try:
       # Code that might raise an exception
   except <ExceptionType>:
       # Handle the exception
   finally:
       # Code to run always
   ```

   **Example:**
   ```python
   try:
       result = 10 / 0
   except ZeroDivisionError:
       print("Cannot divide by zero!")
   finally:
       print("Execution complete.")
   # Output:
   # Cannot divide by zero!
   # Execution complete.
   ```

---

#### 7. **Raising Exceptions**
   - Use the `raise` keyword to manually raise an exception.

   **Syntax:**
   ```python
   raise <ExceptionType>(<message>)
   ```

   **Example:**
   ```python
   def validate_age(age):
       if age < 0:
           raise ValueError("Age cannot be negative!")
       return age

   try:
       validate_age(-5)
   except ValueError as e:
       print(f"Error: {e}")
   # Output: Error: Age cannot be negative!
   ```

---

#### 8. **Custom Exceptions**
   - Define custom exceptions by subclassing `Exception`.

   **Syntax:**
   ```python
   class CustomError(Exception):
       pass
   ```

   **Example:**
   ```python
   class NegativeNumberError(Exception):
       def __init__(self, message="Number cannot be negative!"):
           super().__init__(message)

   def validate_number(num):
       if num < 0:
           raise NegativeNumberError()
       return num

   try:
       validate_number(-10)
   except NegativeNumberError as e:
       print(f"Error: {e}")
   # Output: Error: Number cannot be negative!
   ```

---

#### 9. **Chaining Exceptions**
   - Use `raise from` to chain exceptions and preserve the original traceback.

   **Syntax:**
   ```python
   raise <NewException> from <OriginalException>
   ```

   **Example:**
   ```python
   try:
       10 / 0
   except ZeroDivisionError as e:
       raise ValueError("Invalid operation!") from e
   # Output:
   # ValueError: Invalid operation!
   # The above exception was the direct cause of the following exception:
   # ZeroDivisionError: division by zero
   ```

---

#### 10. **Advanced Exception Handling**
   - **Re-raising Exceptions:** Catch an exception, perform some action, and re-raise it.
   - **Suppressing Exceptions:** Use `contextlib.suppress` to ignore specific exceptions.

   **Example: Re-raising Exceptions**
   ```python
   try:
       10 / 0
   except ZeroDivisionError as e:
       print("Logging the error...")
       raise  # Re-raise the exception
   ```

   **Example: Suppressing Exceptions**
   ```python
   from contextlib import suppress

   with suppress(FileNotFoundError):
       with open("nonexistent_file.txt") as f:
           print(f.read())
   # No error is raised if the file doesn't exist
   ```

---

#### 11. **Best Practices**
   - **Be Specific:** Catch specific exceptions instead of using a bare `except`.
   - **Avoid Silent Failures:** Always log or handle exceptions appropriately.
   - **Use `finally` for Cleanup:** Ensure resources are released using `finally`.
   - **Define Custom Exceptions:** Use custom exceptions for better error handling and readability.

---

#### 12. **Common Mistakes**
   - **Catching Too Broadly:** Avoid catching all exceptions unless absolutely necessary.
   - **Ignoring Exceptions:** Never use a bare `except` without handling the exception.
   - **Overusing Exceptions:** Use exceptions for exceptional cases, not for control flow.

---


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


Classes in Python are the foundation of object-oriented programming (OOP). They allow you to bundle data and functionality together. Below is a detailed cheatsheet on how to create and work with classes, from basic to advanced topics, including attributes, methods, inheritance, access control, and more.

---

### **Basic Concepts**

1. **Class Definition**:
   - **Syntax**:
     ```python
     class <ClassName>:
         # Class body
     ```

2. **Class Object Attributes (Static Variables)**:
   - Class attributes are shared by all instances of the class.
   - These attributes are defined within the class but outside of any methods.
     ```python
     class Car:
         wheels = 4  # Class Object Attribute
     ```

3. **Instance Attributes (Object Attributes)**:
   - Instance attributes are specific to an object and are defined within the `__init__` method.
   - **Syntax**:
     ```python
     class Car:
         def __init__(self, model):
             self.model = model  # Instance Attribute
     ```

4. **Constructor (`__init__` method)**:
   - The constructor is used to initialize the instance attributes when an object is created.
   - **Syntax**:
     ```python
     class Car:
         def __init__(self, model):
             self.model = model
     ```

   - **Creating an Object**:
     ```python
     car1 = Car("Tesla")
     print(car1.model)  # Output: Tesla
     ```

---

### **Methods**: Functions inside a Class

1. **Instance Method**:
   - An instance method takes at least one argument `self`, which refers to the instance of the class.
   - **Syntax**:
     ```python
     class Car:
         def start(self):
             print("Car is starting!")
     ```

2. **Class Method**:
   - Class methods are bound to the class and not the instance. They take a `cls` parameter.
   - **Syntax**:
     ```python
     class Car:
         @classmethod
         def get_class_name(cls):
             return cls.__name__
     ```

   - **Usage**:
     ```python
     print(Car.get_class_name())  # Output: Car
     ```

3. **Static Method**:
   - Static methods do not depend on the class or instance. They are similar to normal functions but reside inside the class.
   - **Syntax**:
     ```python
     class Car:
         @staticmethod
         def get_description():
             return "This is a car."
     ```

   - **Usage**:
     ```python
     print(Car.get_description())  # Output: This is a car.
     ```

---

### **Access Control: Public, Private, and Protected Attributes**

1. **Public Attributes**:
   - Attributes that are accessible from outside the class.
   - **Example**:
     ```python
     class Car:
         def __init__(self, model):
             self.model = model  # Public attribute
     ```

2. **Protected Attributes**:
   - Protected attributes are meant to be used only within the class and its subclasses.
   - Use a single underscore `_` before the attribute name to indicate protected access.
   - **Example**:
     ```python
     class Car:
         def __init__(self, model):
             self._model = model  # Protected attribute
     ```

3. **Private Attributes**:
   - Private attributes are not directly accessible from outside the class.
   - Use a double underscore `__` before the attribute name to indicate private access.
   - **Example**:
     ```python
     class Car:
         def __init__(self, model):
             self.__model = model  # Private attribute
     ```

   - **Accessing Private Attributes**:
     You cannot access private attributes directly:
     ```python
     car = Car("Tesla")
     print(car.__model)  # AttributeError: 'Car' object has no attribute '__model'
     ```

4. **Name Mangling**:
   - Private attributes are not completely hidden; they are "name-mangled" to prevent accidental access.
   - The name of the attribute will be changed internally (e.g., `__model` becomes `_Car__model`).
     ```python
     print(car._Car__model)  # Works but is discouraged
     ```

---

### **Inheritance**: Reusing and Extending Classes

1. **Creating Subclasses**:
   - Inheritance allows one class (child) to inherit attributes and methods from another class (parent).
   - **Syntax**:
     ```python
     class Car:
         def __init__(self, model):
             self.model = model

     class ElectricCar(Car):
         def __init__(self, model, battery_size):
             super().__init__(model)  # Calling parent constructor
             self.battery_size = battery_size
     ```

2. **Calling Parent Constructor (`super()`)**:
   - The `super()` function is used to call the constructor or methods of the parent class.
     ```python
     class ElectricCar(Car):
         def __init__(self, model, battery_size):
             super().__init__(model)
             self.battery_size = battery_size
     ```

3. **Overriding Methods**:
   - Subclasses can override methods of the parent class to provide a custom implementation.
   - **Example**:
     ```python
     class Car:
         def start(self):
             print("Car is starting.")

     class ElectricCar(Car):
         def start(self):
             print("Electric car is starting silently.")
     ```

---

### **Advanced Features**

1. **Multiple Inheritance**:
   - Python supports multiple inheritance, where a class can inherit from more than one class.
   - **Syntax**:
     ```python
     class Engine:
         def start_engine(self):
             print("Engine started.")

     class ElectricCar(Car, Engine):
         pass
     ```

2. **Method Resolution Order (MRO)**:
   - MRO determines the order in which methods are called when using multiple inheritance.
   - Use `class_name.__mro__` to view the MRO of a class.
     ```python
     print(ElectricCar.__mro__)
     ```

3. **Abstract Base Classes (ABC)**:
   - Abstract classes can be created using the `ABC` module to define methods that must be implemented by subclasses.
   - **Syntax**:
     ```python
     from abc import ABC, abstractmethod

     class Vehicle(ABC):
         @abstractmethod
         def start(self):
             pass
     ```

4. **Properties**:
   - Use `@property` to create a getter method and `@<name>.setter` to create a setter for an attribute.
   - **Example**:
     ```python
     class Car:
         def __init__(self, model):
             self._model = model

         @property
         def model(self):
             return self._model

         @model.setter
         def model(self, value):
             if len(value) < 3:
                 raise ValueError("Model name is too short.")
             self._model = value
     ```

---

### **Special Methods (Magic Methods)**

1. **`__init__(self)`**: Constructor (called when an object is created)
   - Already covered.

2. **`__str__(self)`**: String representation of the object (used in `print()` and `str()`).
   - **Example**:
     ```python
     class Car:
         def __init__(self, model):
             self.model = model

         def __str__(self):
             return f"Car model: {self.model}"
     ```

3. **`__repr__(self)`**: Official string representation, used for debugging.
   - **Example**:
     ```python
     class Car:
         def __repr__(self):
             return f"Car(model={self.model})"
     ```

4. **`__eq__(self, other)`**: Equality comparison (`==`).
   - **Example**:
     ```python
     class Car:
         def __eq__(self, other):
             return self.model == other.model
     ```

5. **`__len__(self)`**: Returns the length of an object.
   - **Example**:
     ```python
     class Car:
         def __len__(self):
             return len(self.model)
     ```

---

### **Professional Level**

1. **Metaclasses**:
   - A metaclass defines the behavior of a class. You can control the creation of classes.
   - **Syntax**:
     ```python
     class Meta(type):
         def __new__(cls, name, bases, dct):
             # Modify class before it's created
             return super().__new__(cls, name, bases, dct)

     class Car(metaclass=Meta):
         pass
     ```

2. **Descriptors**:
   - Descriptors allow you to define how attributes of a class are accessed and modified.
   - **Syntax**:
     ```python
     class Descriptor:
         def __get__(self, instance, owner):
             return 'value'

         def __set__(self, instance, value):
             pass

     class Car:
         attr = Descriptor()
     ```

---


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


### **Inheritance in Python**

Inheritance is a key feature of Object-Oriented Programming (OOP) in Python. It allows a class (child class) to inherit the attributes and methods from another class (parent class), enabling reusability and reducing redundancy.

#### **Basic Syntax for Inheritance**

```python
class ParentClass:
    # Attributes and Methods of Parent class
    pass

class ChildClass(ParentClass):
    # Attributes and Methods of Child class
    pass
```

### **Key Concepts in Inheritance**

1. **Parent (Base) Class**: The class whose attributes and methods are inherited by another class.
2. **Child (Derived) Class**: The class that inherits the attributes and methods from the parent class.
3. **`super()` Function**: A built-in function that allows you to call methods from the parent class, especially in the constructor (`__init__`) method.

---

### **Types of Inheritance**

1. **Single Inheritance**:
   - A child class inherits from a single parent class.
   
   ```python
   class Person:
       def __init__(self, name, age):
           self.name = name
           self.age = age

   class Employee(Person):  # Inheriting from Person
       def __init__(self, name, age, staff_num):
           super().__init__(name, age)  # Calling parent constructor
           self.staff_num = staff_num

   emp = Employee("John", 30, 1234)
   print(emp.name)  # Output: John (inherited from Person)
   ```

2. **Multiple Inheritance**:
   - A child class can inherit from more than one class.
   
   ```python
   class Animal:
       def speak(self):
           print("Animal speaks")

   class Bird:
       def fly(self):
           print("Bird flies")

   class Sparrow(Animal, Bird):  # Inheriting from both Animal and Bird
       pass

   sparrow = Sparrow()
   sparrow.speak()  # Output: Animal speaks
   sparrow.fly()    # Output: Bird flies
   ```

3. **Multilevel Inheritance**:
   - A class inherits from a derived class, forming a chain of inheritance.
   
   ```python
   class Person:
       def __init__(self, name):
           self.name = name

   class Employee(Person):
       def __init__(self, name, staff_num):
           super().__init__(name)
           self.staff_num = staff_num

   class Manager(Employee):
       def __init__(self, name, staff_num, department):
           super().__init__(name, staff_num)
           self.department = department

   mgr = Manager("Alice", 1001, "Sales")
   print(mgr.name)        # Output: Alice
   print(mgr.staff_num)   # Output: 1001
   print(mgr.department)  # Output: Sales
   ```

4. **Hierarchical Inheritance**:
   - Multiple classes inherit from a single parent class.
   
   ```python
   class Person:
       def __init__(self, name, age):
           self.name = name
           self.age = age

   class Employee(Person):  # Inheriting from Person
       def __init__(self, name, age, staff_num):
           super().__init__(name, age)
           self.staff_num = staff_num

   class Customer(Person):  # Inheriting from Person
       def __init__(self, name, age, customer_id):
           super().__init__(name, age)
           self.customer_id = customer_id

   emp = Employee("John", 30, 1234)
   customer = Customer("Jane", 28, 5678)

   print(emp.name)  # Output: John
   print(customer.name)  # Output: Jane
   ```

5. **Hybrid Inheritance**:
   - A combination of two or more types of inheritance, such as multiple inheritance and multilevel inheritance together.
   
   ```python
   class A:
       def speak(self):
           print("Class A speaks")

   class B:
       def greet(self):
           print("Class B greets")

   class C(A, B):  # Multiple inheritance
       def hello(self):
           print("Class C says hello")

   class D(C):  # Multilevel inheritance
       pass

   obj = D()
   obj.speak()  # Output: Class A speaks
   obj.greet()  # Output: Class B greets
   obj.hello()  # Output: Class C says hello
   ```

---

### **Using `super()` Function**

- The `super()` function is used to call methods from the parent class.
- It’s especially useful in the `__init__()` method to initialize attributes of the parent class.
  
```python
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

class Employee(Person):
    def __init__(self, name, age, staff_num):
        super().__init__(name, age)  # Calls Person's __init__
        self.staff_num = staff_num

emp = Employee("John", 30, 1234)
print(emp.name)  # Output: John
```

---

### **Overriding Methods**

- A child class can override methods from the parent class to change or extend the functionality.

```python
class Person:
    def speak(self):
        print("I am a person")

class Employee(Person):
    def speak(self):  # Overriding the speak method
        print("I am an employee")

emp = Employee()
emp.speak()  # Output: I am an employee (overridden)
```

---

### **Inheritance and Method Resolution Order (MRO)**

- In Python, when a method is called, the MRO determines the order in which methods are looked up. 
- You can check the MRO using the `mro()` method.

```python
class A:
    def method(self):
        print("A method")

class B(A):
    def method(self):
        print("B method")

class C(B):
    pass

obj = C()
obj.method()  # Output: B method

# Check MRO order
print(C.mro())  # Output: [<class '__main__.C'>, <class '__main__.B'>, <class '__main__.A'>, <class 'object'>]
```

---

### **Accessing Parent Class Methods**

- **Using `super()` to call the parent class method** in the child class:
  
```python
class Animal:
    def sound(self):
        print("Animal sound")

class Dog(Animal):
    def sound(self):
        super().sound()  # Call the parent class method
        print("Woof!")

dog = Dog()
dog.sound()
# Output:
# Animal sound
# Woof!
```

---

### **Abstract Base Class (ABC)**

- You can use an abstract base class (ABC) to define methods that must be implemented in any derived class.
- You can define abstract methods using the `@abstractmethod` decorator.

```python
from abc import ABC, abstractmethod

class Person(ABC):
    @abstractmethod
    def speak(self):
        pass

class Employee(Person):
    def speak(self):
        print("I am an employee")

# emp = Person()  # This will raise an error: TypeError: Can't instantiate abstract class Person with abstract method speak
emp = Employee()
emp.speak()  # Output: I am an employee
```

---

### **Super Important Inheritance Points**

1. **`super()` is used to call parent class methods**, and it's commonly used in the `__init__()` method to initialize the parent class.
2. **Method Overriding** allows a child class to change the behavior of a method inherited from the parent class.
3. **Method Resolution Order (MRO)** determines the order in which the methods are inherited, especially when using multiple inheritance.
4. **Abstract Classes and Methods** allow you to define methods that must be implemented in a child class, ensuring that all subclasses implement certain behaviors.

---

### **Key Inheritance Methods**

- **`issubclass(subclass, parent_class)`**: Checks if a class is a subclass of another.
  
  ```python
  print(issubclass(Employee, Person))  # Output: True
  ```

- **`isinstance(object, class)`**: Checks if an object is an instance of a class.
  
  ```python
  emp = Employee("John", 30, 1234)
  print(isinstance(emp, Person))  # Output: True
  ```

---

### **Final Thoughts**

Inheritance allows for code reusability, and understanding its core concepts like overriding, super, multiple inheritance, and the method resolution order (MRO) is crucial to mastering Python's OOP capabilities. By leveraging inheritance, you can build more efficient and maintainable Python programs.

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
