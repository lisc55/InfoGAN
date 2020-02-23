function obj = coef2object(coef, mu, pc, ev, MM, MB)
if nargin ~= 4 && nargin ~= 6
    error('Inappropriate number of arguments')
end
if nargout ~= 1
    error('One output argument required')
end
n_seg = size(coef, 2);
if nargin == 4 && n_seg > 1
    error('Blending reconstruction requested, but blending parameters missing')
end
n_dim = size(coef, 1);
if n_dim > size(pc, 2)
    error('Too many coefficients.')
end
obj = mu*ones([1 n_seg]) + pc(:,1:n_dim) * (coef .* (ev(1:n_dim)*ones([1 n_seg])) );
if nargin == 4, return; end
n_ver = size(obj,1)/3;
all_vertices = zeros(n_seg*n_ver, 3);
k=0;
for i=1:n_seg
    all_vertices(k+1:k+n_ver, :) = reshape(obj(:,i), 3, n_ver)';
    k = k+n_ver;
end
clear obj k
obj = (MM \ (MB*all_vertices) )';
obj = obj(:);
