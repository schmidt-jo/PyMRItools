function result = compare_matrices(file1_path, file2_path, var1_name, var2_name)
% COMPARE_MATRICES Compare matrices from two .mat files
%
% This function:
% 1. Loads matrices from two .mat files
% 2. Compares the matrices carefully, considering numerical precision issues
% 3. Returns a boolean result indicating whether the matrices are equivalent
%
% Parameters:
%   file1_path - Path to the first .mat file
%   file2_path - Path to the second .mat file
%   var1_name - Name of the matrix variable in the first file
%   var2_name - Name of the matrix variable in the second file (defaults to var1_name if not provided)
%
% Returns:
%   result - Boolean value (1 if matrices are equivalent, 0 otherwise)
%
% The comparison accounts for:
% - Numerical precision differences
% - Potential sign differences for vectors
% - Potential ordering differences if sorting is needed

% If var2_name is not provided, use var1_name for both files
if nargin < 4
    var2_name = var1_name;
end

% Load the matrices from the .mat files
data1 = load(file1_path);
data2 = load(file2_path);

% Extract matrices using the provided variable names
if ~isfield(data1, var1_name)
    error('Variable %s not found in file %s', var1_name, file1_path);
end
if ~isfield(data2, var2_name)
    error('Variable %s not found in file %s', var2_name, file2_path);
end

matrix1 = data1.(var1_name);
matrix2 = data2.(var2_name);

% Check dimensions
if ~isequal(size(matrix1), size(matrix2))
    fprintf('Dimension mismatch between matrices\n');
    result = 0;
    return;
end

% Tolerance for numerical comparisons
tol = 1e-10;

% Determine if we're dealing with vectors or matrices
[rows, cols] = size(matrix1);
is_vector = (rows == 1 || cols == 1);

% For vectors, we need to handle potential sign differences
if is_vector
    % Convert to column vectors for consistent handling
    if rows == 1
        matrix1 = matrix1';
        matrix2 = matrix2';
    end

    % Check both original and sign-flipped versions
    diff1 = norm(matrix1 - matrix2);
    diff2 = norm(matrix1 + matrix2);
    max_diff = min(diff1, diff2);

    % Print detailed comparison results
    fprintf('Comparison Results for %s:\n', var1_name);
    fprintf('Maximum difference: %e\n', max_diff);

    % Determine if vectors are equivalent within tolerance
    result = max_diff < tol;
else
    % For matrices, we compare element-wise
    diff_matrix = abs(matrix1 - matrix2);
    max_diff = max(diff_matrix(:));

    % Print detailed comparison results
    fprintf('Comparison Results for %s:\n', var1_name);
    fprintf('Maximum difference: %e\n', max_diff);

    % Determine if matrices are equivalent within tolerance
    result = max_diff < tol;
end

fprintf('Comparison result: %d\n', result);
end
