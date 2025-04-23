


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