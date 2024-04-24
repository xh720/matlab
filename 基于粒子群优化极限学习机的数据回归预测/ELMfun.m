function fitness = ELMfun(IW, IB, p_train, t_train)

%%  数据的参数
num_size = length(t_train);

%%  交叉验证程序
indices = crossvalind('Kfold', num_size, 5);

for i = 1 : 5
    
    % 获取第i份数据的索引逻辑值
    valid_data = (indices == i);
    
    % 取反，获取第i份训练数据的索引逻辑值
    train_data = ~valid_data;
    
    % 1份测试，4份训练
    pv_train = p_train(:, train_data);
    tv_train = t_train(:, train_data);
    
    pv_valid = p_train(:, valid_data);
    tv_valid = t_train(:, valid_data);

    % 建立模型
    [LW, TF] = elmtrain(pv_train, tv_train, 'sig', 0, IW, IB);

    % 仿真测试
    t_sim = elmpredict(pv_valid, IW, IB, LW, TF);

    % 适应度值
    error(i) = sqrt(sum((t_sim - tv_valid) .^ 2) ./ size(pv_valid, 1));

end

%%  获取适应度
fitness = mean(error);
