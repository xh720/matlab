function fitval = fitcal(pop, indim, hiddennum, p_train, t_train)

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
    error = ELMfun(IW, IB, p_train, t_train);

    % 累计误差
    fitval(i, 1, z) = error; 
end