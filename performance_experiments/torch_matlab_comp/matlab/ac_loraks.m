function ac_loraks(input_path, output_dir)

input_data = load(input_path);
kData = input_data.k_data;
mask = abs(kData) > 1e-10;

% extract sizes
[N1, N2, Nc] = size(kData);
% set neighborhood radius
R = 3;
rank = 50;

% s - matrix
s_matrix = s_operator(kData, N1, N2, Nc, R);

% c - matrix
c_matrix = c_operator(kData, N1, N2, Nc, R);

% get autocalibration region
mac = search_ACS(kData, mask, R);

% eigenvalue decomposition
[U,E] = eig(mac*mac');
[~,idx] = sort(abs(diag(E)),'descend');
U = U(:,idx);

nmm = U(:, 1+rank:end)';

nf = size(nmm,1);
fsz = size(nmm,2)/(2*Nc);
nmm = reshape(nmm,[nf, fsz, 2*Nc]);
% nss_c: complex number representation of the null space
nss_c = reshape(nmm(:,:,1:2:end)+1j*nmm(:,:,2:2:end),[nf, fsz*Nc]);
nmm = reshape(nmm,[nf, fsz*2*Nc]);

% setup filtfilt
fltlen = size(nss_c,2)/Nc;    % filter length
numflt = size(nss_c,1);       % number of filters

% LORAKS kernel is circular.
% Following indices account for circular elements in a square patch
[in1,in2] = meshgrid(-R:R,-R:R);
idx = find(in1.^2+in2.^2<=R^2);
in1 = in1(idx)';
in2 = in2(idx)';

ind = sub2ind([2*R+1, 2*R+1],R+1+in1,R+1+in2);

v_patch = zeros([(2*R+1)*(2*R+1),Nc,numflt],'like',nss_c);
v_patch(ind,:,:) = reshape(permute(nss_c,[2,1]),[fltlen,Nc,numflt]);
v_patch = reshape(v_patch,(2*R+1),(2*R+1),Nc,numflt);

vs_patch = filtfilt_patch(nss_c, "S", N1, N2, Nc, R);
vc_patch = filtfilt_patch(nss_c, "C", N1, N2, Nc, R);

vs_pad = padarray(vs_patch, [N1-1-2*R N2-1-2*R],'post');
vs_shift = circshift(vs_pad,[-4*R-rem(N1,2) -4*R-rem(N2,2)]);
vs = fft2(vs_shift);

vc_pad = padarray(vc_patch, [N1-1-2*R N2-1-2*R], 'post');
vc_shift = circshift(vc_pad,[-2*R -2*R]);
vc = fft2(vc_shift);

data = kData(:);

pad_k = padarray(reshape(data,[N1 N2 Nc]),[2*R, 2*R], 'post');
fft_k = fft2(pad_k);
re_fft_k = repmat(fft_k,[1 1 1 Nc]);
re_conj_fft_k = conj(re_fft_k);

vs_k = sum(vs.*re_conj_fft_k, 3);
vc_k = sum(vc.*re_fft_k, 3);

i_vs_k = ifft2(vs_k);
i_vs_k = i_vs_k(1:N1, 1:N2, :, :);
i_vc_k = ifft2(vc_k);
i_vc_k = i_vc_k(1:N1, 1:N2, :, :);

m = 2 * (i_vc_k - i_vs_k);

% define operators
ZD = @(x) padarray(reshape(x,[N1 N2 Nc]),[2*R, 2*R], 'post');
ZD_H = @(x) x(1:N1,1:N2,:,:);

S.type='()';
S.subs{:} = find(~mask(:));

tmp = zeros([N1 N2 Nc],'like',kData);
A = @(x) data(:) + vect(subsasgn(tmp,S,x));    % embedding missing data operator

Nis = vc;
Nis2 = vs;

M = @(x) 2*subsref(ZD_H(ifft2(squeeze(sum(Nis.*repmat(fft2(ZD(subsasgn(tmp,S,x))),[1 1 1 Nc]),3)))),S) ...
    -2*subsref(ZD_H(ifft2(squeeze(sum(Nis2.*repmat(conj(fft2(ZD(subsasgn(tmp,S,x)))),[1 1 1 Nc]),3)))),S);
b = -2*subsref(ZD_H(ifft2(squeeze(sum(Nis.*repmat(fft2(ZD(data(:))),[1 1 1 Nc]),3)))),S) ...
    +2*subsref(ZD_H(ifft2(squeeze(sum(Nis2.*repmat(conj(fft2(ZD(data(:)))),[1 1 1 Nc]),3)))),S);

tmp_b = subsasgn(tmp, S, b);

[z] = pcg(M, b, 1e-3, 20);
z = A(z);
z = reshape(z, [N1 N2 Nc]);

output_path = fullfile(output_dir, 'matlab_ac_loraks.mat');
save(output_path, ...
    'kData', 'v_patch', 'nmm', 'nss_c', ...
    'c_matrix', 's_matrix', 'mac', ...
    'U', 'vs', 'vc', 'vs_patch', 'vc_patch', ...
    'vc_pad', 'vc_shift', 'vs_pad', 'vs_shift', ...
    'pad_k', 'fft_k', 'vs_k', 'vc_k', ...
    'i_vs_k', 'i_vc_k', 'm', ...
    'Nis', 'Nis2', 'z', 'tmp_b'...
    );

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

% c-operator
function op = c_operator(x, N1, N2, Nc, R)
    x = reshape(x,N1*N2,Nc);

    [in1,in2] = meshgrid(-R:R,-R:R);
    i = find(in1.^2+in2.^2<=R^2);

    in1 = in1(i)';
    in2 = in2(i)';

    patchSize = numel(in1);

    op = zeros(patchSize*Nc,(N1-2*R-even(N1))*(N2-2*R-even(N2)),'like',x);

    k = 0;
    for i = R+1+even(N1):N1-R
        for j = R+1+even(N2):N2-R
            k = k+1;
            ind = sub2ind([N1,N2],i+in1,j+in2);
            op(:,k) = vect(x(ind,:));
        end
    end
end

% zero phase filter
function patch = filtfilt_patch(ncc, opt, N1, N2, Nc, R)
    % Fast computation of zero phase filtering (for alg=4)
    fltlen = size(ncc,2)/Nc;    % filter length
    numflt = size(ncc,1);       % number of filters

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

    cfilt = conj(filtfilt);

    if opt == 'S'       % for S matrix
        ffilt = conj(filtfilt);
    else                % for C matrix
        ffilt = flip(flip(filtfilt,1),2);
    end

    ccfilt = fft2(cfilt,4*R+1, 4*R+1);
    fffilt = fft2(ffilt,4*R+1, 4*R+1);

    patch = ifft2(sum(bsxfun(@times,permute(reshape(ccfilt,4*R+1,4*R+1,Nc,1,numflt),[1 2 4 3 5]) ...
        , reshape(fffilt,4*R+1,4*R+1,Nc,1,numflt)),5));
end

function Nic = filtfilt_nic(patch, opt, N1, N2, R)
    if opt == 'S'       % for S matrix
        Nic = fft2(circshift(padarray(patch, [N1-1-2*R N2-1-2*R],'post'),[-4*R-rem(N1,2) -4*R-rem(N2,2)]));
    else                % for C matrix
        Nic = fft2(circshift(padarray(patch, [N1-1-2*R N2-1-2*R], 'post'),[-2*R -2*R]));
    end
end

function result = even(int)
result = not(rem(int,2));
end

function out = vect( in )
out = in(:);
end

%%
function Mac = search_ACS(data, kMask, R)
% Mac: Matrix for calibration, constructed from ACS
    [N1, N2, Nc] = size(data);

    data = reshape(data,N1*N2,Nc);
    % mask = kMask(:,:,:);
    mask = reshape(kMask,N1*N2,Nc);
    [in1,in2] = meshgrid(-R:R,-R:R);
    i = find(in1.^2+in2.^2<=R^2);

    in1 = in1(i)';
    in2 = in2(i)';

    patchSize = numel(in1);

    Mac = zeros(patchSize,2, Nc, (N1-2*R-even(N1))*(N2-2*R-even(N2)),2,'like',data);

    k = 0;
    for i = R+1+even(N1):N1-R
        for j = R+1+even(N2):N2-R
            k = k+1;
            Ind = sub2ind([N1,N2],i+in1,j+in2);
            Indp = sub2ind([N1,N2],-i+in1+2*ceil((N1-1)/2)+2,-j+in2+2*ceil((N2-1)/2)+2);
            if all(patchSize == sum(mask(Ind, :))) && all(patchSize == sum(mask(Indp, :)))
            % if patchSize == sum(mask(Ind)) && patchSize == sum(mask(Indp))
                tmp = data(Ind,:)-data(Indp,:);
                Mac(:,1,:,k) = real(tmp);
                Mac(:,2,:,k) = -imag(tmp);

                tmp = data(Ind,:)+data(Indp,:);
                Mac(:,1,:,k,2) = imag(tmp);
                Mac(:,2,:,k,2) = real(tmp);
            end
        end
    end
    Mac = reshape(Mac(:,:,:,1:k,:),patchSize*Nc*2,[]);
end

end
