function csp_result = csp(class_A, class_B)

csp_result.sigma_writing = (class_A * class_A')/size(class_A,2);
csp_result.sigma_eating = (class_B * class_B')/size(class_B,2);

% inv(sigma_b) * sigma_A
[csp_result.eigenvectors, csp_result.eigenvalues] = ...
	eig(csp_result.sigma_eating, csp_result.sigma_writing); 

csp_result.w_max3 = csp_result.eigenvectors(:, end:-1:end-2); % max heap
csp_result.w_min3 = csp_result.eigenvectors(:, 3:-1:1); % max heap

csp_result.w = [csp_result.w_max3, csp_result.w_min3];


for i = 1:1001:size(class_A,2)
	s_writing = csp_result.w'*class_A(:, i:i+1000);
	s_eating = csp_result.w'*class_B(:, i:i+1000);
	
	sigma_writing = s_writing*s_writing'/size(s_writing,2);
	sigma_eating = s_eating*s_eating'/size(s_eating,2);
	
	csp_result.var_writing(:,ceil(i/1001)) = diag(sigma_writing);
	csp_result.var_eating(:,ceil(i/1001)) = diag(sigma_eating);
end

csp_result.data = [csp_result.var_writing, csp_result.var_eating];
csp_result.labels = [ones(1,size(csp_result.var_writing,2)), ...
	ones(1,size(csp_result.var_eating,2))*2];