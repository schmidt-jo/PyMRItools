function recon_validation()

mat_vars = load("val_input.mat");
kData = mat_vars.k_data;
mask = mat_vars.mask;
rank = mat_vars.rank;
max_iter = mat_vars.max_num_iter;
reg_lambda = mat_vars.lambda;

k_recon = AC_LORAKS( ...
    kData, mask, ...
    rank, 3, "S", reg_lambda, ...
    4, 1e-5, max_iter ...
    );

% write out data
[filepath,~,~] = fileparts(mfilename('fullpath'));
save(fullfile(filepath, 'val_output.mat'), 'k_recon');

end
