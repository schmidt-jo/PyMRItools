function ac_loraks()

% setup
addpath("/data/u_jschmidt_software/matlab/LORAKS_V2.1/");
% load data - hardcoded path

mat_vars = load("/data/pt_np-jschmidt/code/PyMRItools/tests/experiments/loraks/data/input.mat");
kData = mat_vars.k_data;
mask = mat_vars.mask;
rank = mat_vars.rank;
max_iter = mat_vars.max_num_iter;
reg_lambda = mat_vars.lambda;
num_timer_runs = mat_vars.num_timer_runs;
num_warmup_runs = mat_vars.num_warmup_runs;


% do warmup runs
for i = 1:num_warmup_runs
    k_recon = AC_LORAKS( ...
        kData, mask, ...
        rank, 3, "S", reg_lambda, ...
        4, 1e-5, max_iter ...
        );
    
end
% do timing runs
% profile('-memory','on');
% profile on
t = [];
for i= 1:num_timer_runs

    tic;
    a = AC_LORAKS( ...
        kData, mask, ...
        rank, 3, "S", reg_lambda, ...
        4, 1e-5, max_iter ...
        );
    t(end+1) = toc;

end
% p = profile('info');
% prof = p.FunctionTable;
% write out data
save("/data/pt_np-jschmidt/code/PyMRItools/tests/experiments/loraks/data/output.mat", "t", "k_recon");

end
