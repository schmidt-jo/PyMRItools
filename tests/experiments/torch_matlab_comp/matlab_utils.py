"""
Utility functions for calling MATLAB scripts from Python.

This module provides functions to:
1. Run MATLAB scripts with parameters
2. Perform eigenvalue decomposition using MATLAB
3. Compare matrices between PyTorch and MATLAB
"""

import os
import subprocess
import scipy.io as sio


def run_matlab_script(script_name, script_args=None, script_dir=None, capture_output=True):
    """
    Run a MATLAB script with the given arguments.

    Parameters:
        script_name (str): Name of the MATLAB script file (with or without .m extension)
        script_args (str, optional): Arguments to pass to the MATLAB script
        script_dir (str, optional): Directory containing the MATLAB script
                                   (defaults to 'matlab' subdirectory of current file's directory)
        capture_output (bool, optional): Whether to capture and return the output

    Returns:
        subprocess.CompletedProcess: Result of the subprocess run

    Raises:
        RuntimeError: If the MATLAB script execution fails
    """
    # Ensure script name has .m extension
    if not script_name.endswith('.m'):
        script_name += '.m'

    # Default script directory is 'matlab' subdirectory of current file's directory
    if script_dir is None:
        script_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "matlab")

    # Full path to the script
    script_path = os.path.join(script_dir, script_name)

    # Construct the MATLAB command
    matlab_cmd = f"matlab -nodisplay -nosplash -nodesktop -r \"addpath('{script_dir}'); "
    
    # Extract script name without extension for the function call
    script_func = os.path.splitext(script_name)[0]
    
    # Add function call with arguments if provided
    if script_args:
        matlab_cmd += f"{script_func}({script_args}); "
    else:
        matlab_cmd += f"{script_func}; "
    
    # Add an exit command to close MATLAB after execution
    matlab_cmd += "exit;\""

    # Run the MATLAB command
    result = subprocess.run(matlab_cmd, shell=True, capture_output=capture_output, text=True)
    
    # Check for errors
    if result.returncode != 0:
        if capture_output:
            print(f"MATLAB error: {result.stderr}")
        raise RuntimeError(f"MATLAB script '{script_name}' execution failed")
    
    return result


def compare_matrices(file1_path, file2_path, var_name, output_dir=None, var2_name=None):
    """
    Compare matrices from two .mat files using MATLAB.

    Parameters:
        file1_path (str): Path to the first .mat file
        file2_path (str): Path to the second .mat file
        var_name (str): Name of the variable to compare
        output_dir (str, optional): Directory where to save the comparison result
        var2_name (str, optional): Name of the variable in the second file (if different from var_name)

    Returns:
        bool: True if matrices are equivalent, False otherwise

    Raises:
        RuntimeError: If the MATLAB comparison fails
    """
    # Construct arguments for the MATLAB script
    if var2_name:
        script_args = f"'{file1_path}', '{file2_path}', '{var_name}', '{var2_name}'"
    else:
        script_args = f"'{file1_path}', '{file2_path}', '{var_name}'"
    
    # Add result variable and save command if output_dir is provided
    result_var = f"result_{var_name}"
    if output_dir:
        output_path = os.path.join(output_dir, f"comparison_{var_name}_result.mat")
        script_args += f"; save('{output_path}', '{result_var}')"
    
    # Run the MATLAB script
    run_matlab_script("compare_matrices", script_args)
    
    # Load and return the comparison result if output_dir is provided
    if output_dir:
        comparison_result = sio.loadmat(os.path.join(output_dir, f"comparison_{var_name}_result.mat"))
        return bool(comparison_result[result_var][0][0])
    
    # If no output_dir, we can't return the result (this would require modifying the MATLAB script)
    return None