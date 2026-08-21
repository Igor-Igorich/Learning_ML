import sys
import platform

print(f"Python version: {platform.python_version()}")
print(f"Platform: {platform.system()}")
print(f"Arguments: {sys.argv}")

data = [1, 2, 3, 4, 5]
print(f"Sum: {sum(data)}")