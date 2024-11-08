import sys
print(f"Python executable: {sys.executable}")

try:
    import pandas
    print("Pandas version:", pandas.__version__)
except Exception as e:
    print(f"Error importing pandas: {str(e)}")
    print(f"Python path: {sys.path}")