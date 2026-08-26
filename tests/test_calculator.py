import pytest
from utils.tools import python_calculator, evaluate_math_expression


def test_basic_arithmetic():
    assert evaluate_math_expression("2 + 2") == 4
    assert evaluate_math_expression("10 - 3 * 2") == 4
    assert evaluate_math_expression("20 / 4") == 5.0
    assert evaluate_math_expression("7 // 2") == 3
    assert evaluate_math_expression("10 % 3") == 1
    assert evaluate_math_expression("2 ** 3") == 8
    assert evaluate_math_expression("-5 + 10") == 5
    assert evaluate_math_expression("+5 + 5") == 10


def test_math_functions_and_constants():
    assert evaluate_math_expression("sqrt(144)") == 12.0
    assert evaluate_math_expression("pow(2, 4)") == 16.0
    assert evaluate_math_expression("abs(-42)") == 42
    assert evaluate_math_expression("round(3.14159, 2)") == 3.14
    assert evaluate_math_expression("min(10, 5, 20)") == 5
    assert evaluate_math_expression("max(10, 5, 20)") == 20
    assert evaluate_math_expression("sum([1, 2, 3, 4])") == 10
    assert evaluate_math_expression("sin(0)") == 0.0
    assert evaluate_math_expression("cos(0)") == 1.0
    assert evaluate_math_expression("log(1)") == 0.0
    assert evaluate_math_expression("exp(0)") == 1.0
    assert abs(evaluate_math_expression("pi") - 3.141592653589793) < 1e-6
    assert abs(evaluate_math_expression("e") - 2.718281828459045) < 1e-6


def test_python_calculator_success_string():
    res = python_calculator("sqrt(16) * 10")
    assert "Calculation Result: 40.0" in res


def test_prohibited_attribute_access():
    res = python_calculator("().__class__.__bases__[0]")
    assert "Calculation Error" in res


def test_prohibited_imports_and_builtins():
    res = python_calculator("__import__('os').system('echo pwned')")
    assert "Calculation Error" in res

    res2 = python_calculator("open('/etc/passwd')")
    assert "Calculation Error" in res2

    res3 = python_calculator("exec('x = 1')")
    assert "Calculation Error" in res3


def test_prohibited_lambda_and_statements():
    res = python_calculator("(lambda x: x + 1)(5)")
    assert "Calculation Error" in res


def test_safe_exponent_limits():
    res = python_calculator("10 ** 10 ** 10")
    assert "Calculation Error" in res

    res2 = python_calculator("2 ** 50000")
    assert "Calculation Error" in res2


def test_empty_and_invalid_syntax():
    res = python_calculator("")
    assert "Calculation Error" in res

    res2 = python_calculator("2 + * 3")
    assert "Calculation Error" in res2


def test_natural_language_math_expressions():
    assert evaluate_math_expression("What is the square root of 144 plus 25 multiplied by 4?") == 112.0
    assert evaluate_math_expression("Calculate 100 minus 20 divided by 4") == 95.0
    assert evaluate_math_expression("sqrt of 81 times 2") == 18.0
    assert "112.0" in python_calculator("What is the square root of 144 plus 25 multiplied by 4?")
