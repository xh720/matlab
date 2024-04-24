function [LW, TF] = elmtrain1(new_ptrain, t_train, TF, TYPE, IW, B)

%%  Description
% Input
% P   - Input Matrix of Training Set  (R*Q)  训练输入样本
% T   - Output Matrix of Training Set (S*Q) 训练输出样本
% TF  - Transfer Function: 传递函数，转化函数
%       'sig' for Sigmoidal function (default) S型函数
%       'sin' for Sine function 正弦函数
%       'hardlim' for Hardlim function 硬限制型传递函数
% TYPE - Regression (0,default) or Classification (1)
% IW   - Input Weight Matrix (N*R) 输入权值
% B    - Bias Matrix  (N*1) 偏差
% Output
% LW  - Layer Weight Matrix (N*S)

if size(new_ptrain, 2) ~= size(t_train, 2)  
    error('ELM:Arguments', 'The columns of P and T must be same.');
end

%%  转入分类模式
if TYPE  == 1
    t_train  = ind2vec(t_train);
end

%%  初始化权重
[~, Q] = size(t_train);
BiasMatrix = repmat(B, 1, Q);

%%  计算输出
tempH = IW * new_ptrain + BiasMatrix;

%%  选择激活函数
switch TF
    case 'sig'
        H = 1 ./ (1 + exp(-tempH));
    case 'sin'
        H = sin(tempH);
    case 'hardlim'
        H = hardlim(tempH);
end

%%  伪逆计算权重
LW = pinv(H') * t_train';