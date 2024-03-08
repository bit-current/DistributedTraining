 ```python # Title: Simple Calculator Function

# **Description**
This function takes two numbers as arguments and performs basic arithmetic operations. The supported operations are addition, subtraction, multiplication, and division.

# **Function Definition**
```python
def calculator(num1, num2, operation='+'):
    """Performs addition, subtraction, multiplication or division based on the given input.

    :param num1: First number.
    :type num1: float.
    
    :param num2: Second number.
    :type num2: float.
    
    :param operation: The mathematical operation to be performed. Default is addition (+).
    :type operation: str, optional.
    """

# **Function Body**
try:
    if operation == '+':
        result = num1 + num2
    elif operation == '-':
        result = num1 - num2
    elif operation == '*':
        result = num1 * num2
    elif operation == '/':
        result = num1 / num2
    else:
        raise ValueError("Invalid Operation. Supported Operations are '+', '-', '*', and '/'.")
    
# **Return**
return result
except ZeroDivisionError:
    print("Error: Division by zero is not allowed.")
```

# **Usage Example**
```python calculator(2, 3) # Returns 5.0 calculator(4, -2) # Returns 2.0 calculator(10, 2) # Returns 5.0 calculator(6, 3, '*') # Returns 18.0 calculator(7, 2, '/') # Raises ZeroDivisionError
```