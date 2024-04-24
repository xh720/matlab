% ELM 训练网络
function [LW,TF,TYPE] = elmtrain2(input_train,output_train,N,TF,TYPE,IW,B)

% nargin : 函数输入参数个数
if nargin < 2 
    error('ELM:Arguments','Not enough input arguments.');
end
% 函数输入参数仅有2个 设定默认值

% 获取输入训练样本数的 行 和 列
% R
[R,Z] = size(input_train); %

if nargin < 3  
    N = size(input_train,2);
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
if size(input_train,2) ~= size(output_train,2)  
    error('ELM:Arguments','The columns of P and T must be same.');
end

%  通过ind2vec函数智能进行0 1 分类？不懂
if TYPE  == 1
    output_train  = ind2vec(output_train);
end
[S,Z] = size(output_train);

% % Randomly Generate the Input Weight Matrix
% % 随机生成输入权重矩阵
% IW = rand(N,R) * 2 - 1;
% % Randomly Generate the Bias Matrix
% % 随机生成偏差矩阵
% B = rand(N,1);

% repmat(B,n1,n2);将矩阵B复制n1*n2倍 size(B,2)*n1 size(B,1)*n2
BiasMatrix = repmat(B,1,Z);


% Calculate the Layer Output Matrix H
% 计算输出层矩阵
tempH = IW * input_train + BiasMatrix;
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
LW = pinv(H') * output_train';
