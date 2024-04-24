
function [Alpha_score,Alpha_pos,Convergence_curve]=GWO(SearchAgents_no,Max_iteration,lb,ub,dim,fobj)
%SearchAgents_no 灰狼的种群数量

%%  优化算法初始化
Alpha_pos = zeros(1, dim);  % 初始化Alpha狼的位置
Alpha_score = inf;          % 初始化Alpha狼的目标函数值,将其更改为-inf以解决最大化问题

Beta_pos = zeros(1, dim);   % 初始化Beta狼的位置
Beta_score = inf;           % 初始化Beta狼的目标函数值 ,将其更改为-inf以解决最大化问题

Delta_pos = zeros(1, dim);  % 初始化Delta狼的位置
Delta_score = inf;          % 初始化Delta狼的目标函数值,将其更改为-inf以解决最大化问题

%%  初始化搜索狼群的位置
Positions = initialization(SearchAgents_no, dim, ub, lb);

%%  用于记录迭代曲线
Convergence_curve = zeros(1, Max_iteration);
%%  循环计数器
iter = 0;

%%  优化算法主循环
while iter < Max_iteration           % 对迭代次数循环
    for i = 1 : size(Positions, 1)   % 遍历每个狼

        % 返回超出搜索空间边界的搜索狼群
        % 若搜索位置超过了搜索空间，需要重新回到搜索空间
        Flag4ub = Positions(i, :) > ub;
        Flag4lb = Positions(i, :) < lb;

        % 若狼的位置在最大值和最小值之间，则位置不需要调整，若超出最大值，最回到最大值边界
        % 若超出最小值，最回答最小值边界
        Positions(i, :) = (Positions(i, :) .* (~(Flag4ub + Flag4lb))) + ub .* Flag4ub + lb .* Flag4lb;   

        % 计算适应度函数值
%         Positions(i, 2) = round(Positions(i, 2));
%         fitness = fical(Positions(i, :));
          fitness = fobj(Positions(i, :));
        % 更新 Alpha, Beta, Delta
        if fitness < Alpha_score           % 如果目标函数值小于Alpha狼的目标函数值
            Alpha_score = fitness;         % 则将Alpha狼的目标函数值更新为最优目标函数值
            Alpha_pos = Positions(i, :);   % 同时将Alpha狼的位置更新为最优位置
        end

        if fitness > Alpha_score && fitness < Beta_score   % 如果目标函数值介于于Alpha狼和Beta狼的目标函数值之间
            Beta_score = fitness;                          % 则将Beta狼的目标函数值更新为最优目标函数值
            Beta_pos = Positions(i, :);                    % 同时更新Beta狼的位置
        end

        if fitness > Alpha_score && fitness > Beta_score && fitness < Delta_score  % 如果目标函数值介于于Beta狼和Delta狼的目标函数值之间
            Delta_score = fitness;                                                 % 则将Delta狼的目标函数值更新为最优目标函数值
            Delta_pos = Positions(i, :);                                           % 同时更新Delta狼的位置
        end

    end

    % 线性权重递减
    wa = 2 - iter * ((2) / Max_iteration);    

    % 更新搜索狼群的位置
    for i = 1 : size(Positions, 1)      % 遍历每个狼
        for j = 1 : size(Positions, 2)  % 遍历每个维度

            % 包围猎物，位置更新
            r1 = rand; % r1 is a random number in [0,1]
            r2 = rand; % r2 is a random number in [0,1]

            A1 = 2 * wa * r1 - wa;   % 计算系数A，Equation (3.3)
            C1 = 2 * r2;             % 计算系数C，Equation (3.4)

            % Alpha 位置更新
            D_alpha = abs(C1 * Alpha_pos(j) - Positions(i, j));   % Equation (3.5)-part 1
            X1 = Alpha_pos(j) - A1 * D_alpha;                     % Equation (3.6)-part 1

            r1 = rand; % r1 is a random number in [0,1]
            r2 = rand; % r2 is a random number in [0,1]

            A2 = 2 * wa * r1 - wa;   % 计算系数A，Equation (3.3)
            C2 = 2 *r2;              % 计算系数C，Equation (3.4)

            % Beta 位置更新
            D_beta = abs(C2 * Beta_pos(j) - Positions(i, j));    % Equation (3.5)-part 2
            X2 = Beta_pos(j) - A2 * D_beta;                      % Equation (3.6)-part 2       

            r1 = rand;  % r1 is a random number in [0,1]
            r2 = rand;  % r2 is a random number in [0,1]

            A3 = 2 *wa * r1 - wa;     % 计算系数A，Equation (3.3)
            C3 = 2 *r2;               % 计算系数C，Equation (3.4)

            % Delta 位置更新
            D_delta = abs(C3 * Delta_pos(j) - Positions(i, j));   % Equation (3.5)-part 3
            X3 = Delta_pos(j) - A3 * D_delta;                     % Equation (3.5)-part 3

            % 位置更新
            Positions(i, j) = (X1 + X2 + X3) / 3;                 % Equation (3.7)

        end
    end

    % 更新迭代器
    iter = iter + 1;    
    Convergence_curve(iter) = Alpha_score;
    disp(['第',num2str(iter),'次迭代'])
    disp(['current iteration is: ',num2str(iter), ', best fitness is: ', num2str(Alpha_score)]);
end

%%  记录最佳参数
% best_lr = Alpha_pos(1, 1);
% best_hd = Alpha_pos(1, 2);
% best_l2 = Alpha_pos(1, 3);
end