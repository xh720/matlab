function fitval1 = fitcal1(pop, indim, hiddennum, new_ptrain, t_train)

%%  得到种群规模
[x, ~, z] = size(pop);

for i = 1 : x

    % 提取输入权重
    for j = 1 : hiddennum
        IW(j, :) = pop(i, ((j - 1) * indim + 1) : j * indim, z);
    end

    % 得到隐藏层偏置
    IB = pop(i, end -  hiddennum + 1: end, z)';
    
    % 得到误差
    error = ELMfun1(IW, IB, new_ptrain, t_train);

    % 累计误差
    fitval1(i, 1, z) = error; 
end