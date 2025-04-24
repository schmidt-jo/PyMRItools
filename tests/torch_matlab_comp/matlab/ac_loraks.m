function ac_loraks(input_path, output_dir)

input_data = load(input_path);
kData = input_data.k_data;

% extract sizes
[N1, N2, Nc] = size(kData);
% set neighborhood radius
R = 3;

% s - matrix
s_matrix = s_operator(kData, N1, N2, Nc, R);

% eigenvalue decomposition
[U,E] = eig(s_matrix*s_matrix');
[~,idx] = sort(abs(diag(E)),'descend');
U = U(:,idx);

nmm = U(:, 1+rank:end)';

nmm = reshape(nmm,[nf, fsz, 2*Nc]);
% nss_c: complex number representation of the null space
nss_c = reshape(nmm(:,:,1:2:end)+1j*nmm(:,:,2:2:end),[nf, fsz*Nc]);
nmm = reshape(nmm,[nf, fsz*2*Nc]);

%% setup filtfilt
fltlen = size(nss_c,2)/Nc;    % filter length
numflt = size(nss_c,1);       % number of filters

% LORAKS kernel is circular.
% Following indices account for circular elements in a square patch
[in1,in2] = meshgrid(-R:R,-R:R);
idx = find(in1.^2+in2.^2<=R^2);
in1 = in1(idx)';
in2 = in2(idx)';

ind = sub2ind([2*R+1, 2*R+1],R+1+in1,R+1+in2);

filtfilt = zeros([(2*R+1)*(2*R+1),Nc,numflt],'like',ncc);
filtfilt(ind,:,:) = reshape(permute(ncc,[2,1]),[fltlen,Nc,numflt]);
filtfilt = reshape(filtfilt,(2*R+1),(2*R+1),Nc,numflt);

output_path = fullfile(output_dir, 'matlab_loraks_nmm.mat');
save(output_path, 'filtfilt', 'nmm', 'nss_c', 's_matrix', 'U');


% s - operator
function op = s_operator(x, N1, N2, Nc, R)
    x = reshape(x,N1*N2,Nc);

    [in1,in2] = meshgrid(-R:R,-R:R);
    i = find(in1.^2+in2.^2<=R^2);

    in1 = in1(i)';
    in2 = in2(i)';

    patchSize = numel(in1);

    result = zeros(patchSize,2, Nc, (N1-2*R-even(N1))*(N2-2*R-even(N2))*2,'like',x);

    k = 0;
    for i = R+1+even(N1):N1-R
        for j = R+1+even(N2):N2-R
            k = k+1;
            Ind = sub2ind([N1,N2],i+in1,j+in2);
            Indp = sub2ind([N1,N2],-i+in1+2*ceil((N1-1)/2)+2,-j+in2+2*ceil((N2-1)/2)+2);

            tmp = x(Ind,:)-x(Indp,:);
            result(:,1,:,k) = real(tmp);
            result(:,2,:,k) = -imag(tmp);

            tmp = x(Ind,:)+x(Indp,:);
            result(:,1,:,k+end/2) = imag(tmp);
            result(:,2,:,k+end/2) = real(tmp);
        end
    end

    op = reshape(result, patchSize*Nc*2,(N1-2*R-even(N1))*(N2-2*R-even(N2))*2);
end


