

# Have a good day

def my_finc1(x):
    return x**2


def numerical_derivative(f,x):
    delta_x = 1e-4
    return (f(x+delta_x)-f(x-delta_x))/(2*delta_x)


result = numerical_derivative(my_finc1, 123)
print("Result=", result)
