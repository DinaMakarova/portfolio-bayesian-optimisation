import subprocess
import sys

def run_function_script(function_number):
    """Executes the script for a given function number and prints its output."""
    script_name = f'func_{function_number}.py'
    print("\n" + "#" * 70)
    print(f"### Running Analysis for Function {function_number} using '{script_name}' ###")
    print("#" * 70)
    
    try:
        # We use sys.executable to ensure the script runs with the same Python interpreter
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=True,
            text=True,
            check=True  # This will raise an exception if the script returns a non-zero exit code
        )
        print(result.stdout)
        if result.stderr:
            print("--- Errors/Warnings ---")
            print(result.stderr)
            
    except FileNotFoundError:
        print(f"ERROR: The script '{script_name}' was not found in the current directory.")
    except subprocess.CalledProcessError as e:
        print(f"ERROR: An error occurred while running '{script_name}'.")
        print("--- STDOUT ---")
        print(e.stdout)
        print("--- STDERR ---")
        print(e.stderr)

if __name__ == "__main__":
    for i in range(1, 9):
        run_function_script(i)