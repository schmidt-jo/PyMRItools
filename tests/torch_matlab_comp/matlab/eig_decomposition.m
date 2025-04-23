function eig_decomposition(input_path, output_dir)
% EIG_DECOMPOSITION Perform eigenvalue decomposition on a matrix
%
% This function:
% 1. Loads a matrix from a .mat file
% 2. Performs eigenvalue decomposition using MATLAB's eig function
% 3. Saves the results to a .mat file
%
% Parameters:
%   input_path - Path to the input .mat file containing the matrix
%   output_dir - Directory where the output will be saved
%
% The input .mat file should contain a variable named 'matrix'.
% The output .mat file will contain 'eigenvalues' and 'eigenvectors'.

% Load the input matrix
input_data = load(input_path);
matrix = input_data.matrix;

[V, D] = eig(matrix);

% Extract eigenvalues from the diagonal matrix D
eigenvalues_unsorted = diag(D);

% Sort eigenvalues in descending order and get the sorting indices
[sorted_eigenvalues, sort_indices] = sort(eigenvalues_unsorted, 'descend');

% Reorder the eigenvectors accordingly
eigenvectors = V(:, sort_indices);

% Reconstruct the diagonal matrix of sorted eigenvalues
eigenvalues = sorted_eigenvalues;

% Save the results
output_path = fullfile(output_dir, 'matlab_eig_result.mat');
save(output_path, 'eigenvalues', 'eigenvectors');

fprintf('Eigenvalue decomposition completed. Results saved to: %s\n', output_path);
end