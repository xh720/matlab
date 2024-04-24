%%  清空环境变量
warning off             % 关闭报警信息
close all               % 关闭开启的图窗
clear                   % 清空变量
clc                     % 清空命令行

%%  导入数据
res = xlsread('F:\大四\数据集.xlsx');

%%  数据分析
num_size = 0.8;                              % 训练集占数据集比例
outdim = 1;                                  % 最后一列为输出
num_samples = size(res, 1);                  % 样本个数
res = res(randperm(num_samples), :);         % 打乱数据集（不希望打乱时，注释该行）
num_train_s = round(num_size * num_samples); % 训练集样本个数
f_ = size(res, 2) - outdim;                  % 输入特征维度

%%  划分训练集和测试集
P_train = res(1: num_train_s, 1: f_)';
T_train = res(1: num_train_s, f_ + 1: end)';
M = size(P_train, 2);

P_test = res(num_train_s + 1: end, 1: f_)';
T_test = res(num_train_s + 1: end, f_ + 1: end)';
N = size(P_test, 2);

%%  数据归一化
[p_train, ps_input] = mapminmax(P_train, 0, 1);
p_test = mapminmax('apply', P_test, ps_input);

[t_train, ps_output] = mapminmax(T_train, 0, 1);
t_test = mapminmax('apply', T_test, ps_output);

%%  参数初始化设置
hiddennum = 50;                   % 隐藏层维度 BP神经网络中通常小于输入节点数大于输出节点数
indim     = size(p_train, 1);     % 输入层维度 7
outdim    = size(t_train, 1);     % 输出层维度 1

%%  粒子群参数设置
c1      = 1.45;         % 学习因子
c2      = 1.64;         % 学习因子
iternum =   30;         % 种群规模
Pmax    =    1;         % 种群最大值
Pmin    =   -1;         % 种群最小值
Vmax    =  0.5;         % 最大速度
Vmin    = -0.5;         % 最小速度
wmax    =  0.8;         % 惯性因子起始值
wmin    =  0.1;         % 惯性因子终止值
itmax   =  100;         % 种群更新次数  
minerr  = 1e-6;         % 误差阈值

%%  种群初始化

% 给定适应度值的存储空间
MinFit  = [];
BestFit = [];

% 初始化权重惯性因子初始化
for iter = 1: itmax
    W(iter) = wmax - ((wmax - wmin) / itmax) * iter; 
end

% 初始化待优化粒子数目  输入层节点数加1再乘以隐藏层节点数
D_nodes = (indim + 1) * hiddennum;  %神经网络中所有权重和偏置的数量

% 初始化种群和速度
Pop = 2 * rand(iternum, D_nodes, 1) - 1; % ？
V   = 2 * rand(iternum, D_nodes, 1) - 1;

%%  初始化粒子群各参数
fitness = fitcal(Pop, indim, hiddennum, p_train, t_train);  % 初始化适应度值
fvrec(:, 1, 1) = fitness(:, 1, 1);                          % 存储适应度值
[C, I] = min(fitness(:, 1, 1));                             % 得到最佳适应度值
MinFit  = [MinFit , C];                                     % 存储个体最佳适应度值
BestFit = [BestFit, C];                                     % 存储全部最佳适应度值
L(:, 1, 1) = fitness(:, 1, 1);                              % 
B(1, 1, 1) = C;                                             % 当前适应度值
gbest(1, :, 1) = Pop(I, :, 1);                              % 储存最佳种群

for i = 1 : iternum
    zbest(i, :, 1) = Pop(1, :, 1);                          % 全局最佳种群
    pbest(i, :, 1) = Pop(i, :, 1);                          % 个体最佳种群
end

%%  得到当前速度值
V(:, :, 2) = W(1) * V(:, :, 1) + c1 * rand * (pbest(:, :, 1) - Pop(:, :, 1)) ...
              + c2 * rand * (zbest(:, :, 1) - Pop(:, :, 1));

%%  设置速度边界
for ni = 1 : iternum
    for di = 1 : D_nodes

        if V(ni, di, 2) > Vmax
            % 设置上边界
            V(ni, di, 2) = Vmax;
        elseif V(ni, di, 2) < Vmin
            % 设置下边界
            V(ni, di, 2) = Vmin;
        end

    end
end

%%  种群更新
Pop(:, :, 2) = Pop(:, :, 1) + V(:, :, 2);

%%  设置种群边界
for ni = 1 : iternum
    for di = 1 : D_nodes

        if Pop(ni, di, 2) > Pmax
            % 种群上界
            Pop(ni, di, 2) = Pmax;
        elseif V(ni, di, 2) < -Pmin
            % 种群下界
            Pop(ni, di, 2) = -Pmin;
        end

    end
end

%%  迭代优化
for j = 2 : itmax

    % 得到适应度值
    fitness = fitcal(Pop, indim, hiddennum, p_train, t_train); 

    % 参数更新
    fvrec(:, 1, j) = fitness(:, 1, j);    %将当前适应度值存储在fvrec矩阵的第一列中
    [C, I] = min(fitness(:, 1, j));
    MinFit = [MinFit, C];                   % 存储个体最佳适应度值
    BestFit = [BestFit, min(MinFit)];       % 存储最佳全局适应度值
    L(:, 1, j) = fitness(:, 1, j);          % 累计适应度值
    B(1, 1, j) = C;                         % 存储最小误差阈值
    gbest(1, :, j) = Pop(I, :, j);          % 得到全局最佳种群
    [C, I] = min(B(1, 1, :));               % 最小误差及索引
    
    % 得到全局最佳种群
    if B(1, 1, j) <= C
        gbest(1, :, j) = gbest(1, :, j);
    else
        gbest(1, :, j) = gbest(1, :, I);
    end
    
    % 若误差（适应度值）小于阈值，跳出循环
    if (C <= minerr)
        break
    end

    % 若迭代次数超出最大迭代次数，跳出循环
    if (j >= itmax)
        break
    end

    % 存储全局最佳种群
    for p = 1 : iternum
        zbest(p, :, j) = gbest(1, :, j);
    end

    % 得到个体最佳种群
    for i = 1 : iternum
        [C, I] = min(L(i, 1, :));
        if L(i, 1, j) <= C
            pbest(i, :, j) = Pop(i, :, j);
        else
            pbest(i, :, j) = Pop(i, :, I);
        end
    end

    % 更新速度
    V(:, :, j + 1) = W(j) * V(:, :, j) + c1 * rand * (pbest(:, :, j) - ...
        Pop(:, :, j)) + c2 * rand*(zbest(:, :, j) - Pop(:, :, j));
    
    % 速度边界设置
    for ni = 1 : iternum
        for di = 1 : D_nodes

            if V(ni, di, j + 1) > Vmax
                % 设置速度上限
               V(ni, di, j + 1) = Vmax;
            elseif V(ni, di, j + 1) < Vmin
                % 设置速度下限
               V(ni, di, j + 1) = Vmin;
            end

        end
    end

    % 更新种群
    Pop(:, :, j + 1) = Pop(:, :, j) + V(:, :, j + 1);
    
    % 种群边界设置
    for ni = 1 : iternum
        for di = 1 : D_nodes

            if Pop(ni, di, j + 1) > Pmax
               % 超出种群边界上限
               Pop(ni, di, j + 1) = Pmax;
            elseif V(ni, di, j + 1) < Pmin
               % 超出种群边界下限
               Pop(ni, di, j + 1) = Pmin;
            end

        end
    end 

end

%%  提取最优权重
for t = 1 : hiddennum
    IWbest(t, :) = gbest(1, ((t - 1) * indim + 1) : t * indim, j);
end
IBbest = gbest(1, end - hiddennum + 1 : end, j)';

%%  建立模型
[LW, TF] = elmtrain(p_train, t_train, 'sig', 0, IWbest, IBbest);

%%  仿真预测
t_sim1 = elmpredict(p_train, IWbest, IBbest, LW, TF);
t_sim2 = elmpredict(p_test , IWbest, IBbest, LW, TF);

%%  数据反归一化
T_sim1 = mapminmax('reverse', t_sim1, ps_output);
T_sim2 = mapminmax('reverse', t_sim2, ps_output);

%%  均方根误差
RMSE1 = sqrt(sum((T_sim1 - T_train).^2) ./ M);
RMSE2 = sqrt(sum((T_sim2 - T_test ).^2) ./ N);

%%  绘图
figure
plot(1: M, T_train, 'r-*', 1: M, T_sim1, 'b-o', 'LineWidth', 1)
legend('真实值', '预测值')
xlabel('预测样本')
ylabel('预测结果')
string = {'训练集预测结果对比'; ['RMSE=' num2str(RMSE1)]};
title(string)
xlim([1, M])
grid

figure
plot(1: N, T_test, 'r-*', 1: N, T_sim2, 'b-o', 'LineWidth', 1)
legend('真实值', '预测值')
xlabel('预测样本')
ylabel('预测结果')
string = {'测试集预测结果对比'; ['RMSE=' num2str(RMSE2)]};
title(string)
xlim([1, N])
grid

%%  误差曲线迭代图
figure;
plot(1: length(BestFit), BestFit, 'LineWidth', 1.5);
xlabel('粒子群迭代次数');
ylabel('适应度值');
xlim([1, length(BestFit)])
string = {'模型迭代误差变化'};
title(string)
grid on

%%  相关指标计算
%  R2
R1 = 1 - norm(T_train - T_sim1)^2 / norm(T_train - mean(T_train))^2;
R2 = 1 - norm(T_test  - T_sim2)^2 / norm(T_test  - mean(T_test ))^2;

disp(['训练集数据的R2为：', num2str(R1)])
disp(['测试集数据的R2为：', num2str(R2)])

%  MAE
mae1 = sum(abs(T_sim1 - T_train)) ./ M ;
mae2 = sum(abs(T_sim2 - T_test )) ./ N ;

disp(['训练集数据的MAE为：', num2str(mae1)])
disp(['测试集数据的MAE为：', num2str(mae2)])

%  MBE
mbe1 = sum(T_sim1 - T_train) ./ M ;
mbe2 = sum(T_sim2 - T_test ) ./ N ;

disp(['训练集数据的MBE为：', num2str(mbe1)])
disp(['测试集数据的MBE为：', num2str(mbe2)])

%MAPE
error1=T_sim1-T_train;
error2=T_sim2-T_test;
MAPE1=mean(abs(error1./T_train));
MAPE2=mean(abs(error2./T_test));

disp(['训练集预测准确率为：',num2str(100-MAPE1*100),'%'])
disp(['测试集预测准确率为：',num2str(100-MAPE2*100),'%'])

ms_pso = abs(T_sim2-T_test)./T_test;
%% %%计算准确率  
predict_value=T_sim2;
true_value=T_test;
correct = 0;  
total = length(predict_value);  
% 遍历每一个样本  
for i = 1:total  
    if predict_value(i) <= true_value(i)+  true_value(i)*0.02 && predict_value(i) >= true_value(i)- true_value(i)*0.02
        correct = correct + 1;  
    end  
end   

accuracy = correct / total;
disp(['误差准确率为：',num2str(accuracy)])

pre2_pso=T_sim2;
r2_pso=R2;
mae_pso=mae2;
rmse_pso=RMSE2;

save PSO.mat pre2_pso r2_pso mae_pso rmse_pso ms_pso T_test