%X表示自变量集合
clc;clear;
res = xlsread('F:\大四\数据集.xlsx');
rng('default')
%%  数据分析
num_size = 0.8;                              % 训练集占数据集比例
t = 1;                                       % 输出变量最后一列为输出
num_samples = size(res, 1);                  % 样本个数
res = res(randperm(num_samples), :);         % 打乱数据集（不希望打乱时，注释该行）
num_train_s = round(num_size * num_samples); % 训练集样本个数
x = size(res, 2) - t;                        % 输入变量特征维度

%% 划分训练集和测试集

P_train = res(1: num_train_s, 1: x)';     %训练输入
T_train = res(1: num_train_s, x + 1: end)';  %训练输出
m = size(P_train, 2);

P_test = res(num_train_s + 1: end, 1: x)';
T_test = res(num_train_s + 1: end, x + 1: end)';
n = size(P_test, 2);

inputnum=size(P_train,1);       % 输入层神经元个数
outputnum=size(T_train,1);      % 输出层神经元个数=偏置个数
hiddennum=35; % 隐含层神经元个数
w1num=inputnum*hiddennum; % 输入层到隐层的w个数
w2num=outputnum*hiddennum;% 隐含层到输出层的w个数
N=w1num+hiddennum+w2num+outputnum;%自变量的总数

%%  数据归一化
[p_train, ps_input] = mapminmax(P_train, 0, 1);
p_test = mapminmax('apply', P_test, ps_input);

[t_train, ps_output] = mapminmax(T_train, 0, 1);
t_test = mapminmax('apply', T_test, ps_output);

%% % 未优化直接运行ELM结果
[T_train,T_sim1,T_test,T_sim01]=ELM(P_train,T_train,P_test,T_test,hiddennum);

%% 定义遗传算法参数
NIND=30;        %个体数目
MAXGEN=200;     %最大遗传代数
PRECI=10;       %变量的二进制位数
GGAP=0.95;      %代沟
px=0.8;         %交叉概率
pm=0.01;        %变异概率
trace=zeros(N+1,MAXGEN);                        %寻优结果的初始值

FieldD=[repmat(PRECI,1,N);repmat([-0.5;0.5],1,N);repmat([1;0;1;1],1,N)];                      %区域描述器
Chrom=crtbp(NIND,PRECI*N);                      %初始种群
%% 优化
gen=0;                                 %代计数器
X=bs2rv(Chrom,FieldD);                 %计算初始种群的十进制转换
ObjV=Objfun(X,p_train,t_train,hiddennum,p_test,t_test);        %计算目标函数值
while gen<MAXGEN
   fprintf('%d\n',gen);
   FitnV=ranking(ObjV);                              %分配适应度值
   SelCh=select('sus',Chrom,FitnV,GGAP);              %选择
   SelCh=recombin('xovsp',SelCh,px);                  %重组
   SelCh=mut(SelCh,pm);                               %变异
   X=bs2rv(SelCh,FieldD);               %子代个体的十进制转换
   ObjVSel=Objfun(X,p_train,t_train,hiddennum,p_test,t_test);             %计算子代的目标函数值
   [Chrom,ObjV]=reins(Chrom,SelCh,1,1,ObjV,ObjVSel); %重插入子代到父代，得到新种群
   X=bs2rv(Chrom,FieldD);
   gen=gen+1;                                             %代计数器增加
   %获取每代的最优解及其序号，Y为最优解,I为个体的序号
   [Y,I]=min(ObjV);
   trace(1:N,gen)=X(I,:);                       %记下每代的最优值
   trace(end,gen)=Y;                               %记下每代的最优值
end

%% 画进化图
figure(3);
P0 = plot(1:MAXGEN,trace(end,:));
grid on
xlabel('遗传代数')
ylabel('误差的变化')
title('进化过程')
bestX=trace(1:end-1,end);
bestErr=trace(end,end);
fprintf(['最优初始权值和阈值:\nX=',num2str(bestX'),'\n最小误差err=',num2str(bestErr),'\n'])
set(P0,'LineWidth',1.5);       %| 设置图形线宽

%% 优化之后的ELM
[acc,T_train2,T_sim2,T_test2,T_sim02]=GAELM(bestX,P_train,T_train,hiddennum,P_test,T_test);

%% 绘图

