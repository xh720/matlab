% ELM 训练网络
function [LW,TF,TYPE] = elmtrain(P,T,N,TF,TYPE,IW,B)

% nargin : 函数输入参数个数
if nargin < 2 
    error('ELM:Arguments','Not enough input arguments.');
end
% 函数输入参数仅有2个 设定默认值

% 获取输入训练样本数的 行 和 列
% R
[R,Q] = size(P); %

if nargin < 3  
    N = size(P,2);
end
% 函数输入参数仅有3个，加入默认函数 设定默认值
if nargin < 4 
    TF = 'sig';
end
% 函数输入参数仅有4个，设定函数类型 Regression (0,default) or Classification (1) 设定默认值
if nargin < 5 
    TYPE = 0;
end   
if nargin < 6 
    IW = rand(N,R) * 2 - 1;
end   
if nargin < 7 
    B = rand(N,1);
end   
% ~= 不等于 检验样本数是否一致
if size(P,2) ~= size(T,2)  
    error('ELM:Arguments','The columns of P and T must be same.');
end

%  通过ind2vec函数智能进行0 1 分类？不懂
if TYPE  == 1
    T  = ind2vec(T);
end
[S,Q] = size(T);

% % Randomly Generate the Input Weight Matrix
% % 随机生成输入权重矩阵
% IW = rand(N,R) * 2 - 1;
% % Randomly Generate the Bias Matrix
% % 随机生成偏差矩阵
% B = rand(N,1);

% repmat(B,n1,n2);将矩阵B复制n1*n2倍 size(B,2)*n1 size(B,1)*n2
BiasMatrix = repmat(B,1,Q);


% Calculate the Layer Output Matrix H
% 计算输出层矩阵
tempH = IW * P + BiasMatrix;
switch TF
    case 'sig'
        H = 1 ./ (1 + exp(-tempH));
    case 'sin'
        H = sin(tempH);
    case 'hardlim'
        H = hardlim(tempH);
end


% Calculate the Output Weight Matrix
% 计算输出层权值矩阵
LW = pinv(H') * T';


%  相关注释
% ELMTRAIN Create and Train a Extreme Learning Machine
% Syntax 语法
% [IW,B,LW,TF,TYPE] = elmtrain(P,T,N,TF,TYPE)
% Description 描述
% Input
% P   - Input Matrix of Training Set  (R*Q)  训练输入样本
% T   - Output Matrix of Training Set (S*Q) 训练输出样本
% N   - Number of Hidden Neurons (default = Q) 隐含层节点数
% TF  - Transfer Function: 传递函数，转化函数
%       'sig' for Sigmoidal function (default) S型函数
%       'sin' for Sine function 正弦函数
%       'hardlim' for Hardlim function 硬限制型传递函数
% TYPE - Regression (0,default) or Classification (1)
% Output
% IW  - Input Weight Matrix (N*R) 输入权值
% B   - Bias Matrix  (N*1) 偏差
% LW  - Layer Weight Matrix (N*S)
% Example
% Regression:
% [IW,B,LW,TF,TYPE] = elmtrain(P,T,20,'sig',0)
% Y = elmtrain(P,IW,B,LW,TF,TYPE)
% Classification
% [IW,B,LW,TF,TYPE] = elmtrain(P,T,20,'sig',1)
% Y = elmtrain(P,IW,B,LW,TF,TYPE)
% See also ELMPREDICT
% Yu Lei,11-7-2010
% Copyright www.matlabsky.com
% $Revision:1.0 $