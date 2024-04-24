
function [IWbest1,IBbest1]=pso(hiddennum,new_ptrain,t_train)
indim1     = size(new_ptrain, 1);

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
D_nodes = (indim1 + 1) * hiddennum;  %神经网络中所有权重和偏置的数量

% 初始化种群和速度
Pop = 2 * rand(iternum, D_nodes, 1) - 1; % ？
V   = 2 * rand(iternum, D_nodes, 1) - 1;

%%  初始化粒子群各参数
fitness1 = fitcal1(Pop, indim1, hiddennum, new_ptrain, t_train);  % 初始化适应度值
fvrec(:, 1, 1) = fitness1(:, 1, 1);                          % 存储适应度值
[C, I] = min(fitness1(:, 1, 1));                             % 得到最佳适应度值
MinFit  = [MinFit , C];                                     % 存储个体最佳适应度值
BestFit = [BestFit, C];                                     % 存储全部最佳适应度值
L(:, 1, 1) = fitness1(:, 1, 1);                              % 
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
    fitness1 = fitcal1(Pop, indim1, hiddennum, new_ptrain, t_train); 

    % 参数更新
    fvrec(:, 1, j) = fitness1(:, 1, j);    %将当前适应度值存储在fvrec矩阵的第一列中
    [C, I] = min(fitness1(:, 1, j));
    MinFit = [MinFit, C];                   % 存储个体最佳适应度值
    BestFit = [BestFit, min(MinFit)];       % 存储最佳全局适应度值
    L(:, 1, j) = fitness1(:, 1, j);          % 累计适应度值
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
    IWbest1(t, :) = gbest(1, ((t - 1) * indim1 + 1) : t * indim1, j);
end
IBbest1 = gbest(1, end - hiddennum + 1 : end, j)';