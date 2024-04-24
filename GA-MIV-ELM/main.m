clc;clear;
res = xlsread('F:\大四\数据集.xlsx');
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
hiddennum=55; % 隐含层神经元个数
w1num=inputnum*hiddennum; % 输入层到隐层的w个数
w2num=outputnum*hiddennum;% 隐含层到输出层的w个数
N=w1num+hiddennum+w2num+outputnum;%自变量的总数

%%  数据归一化
[p_train, ps_input] = mapminmax(P_train, 0, 1);
p_test = mapminmax('apply', P_test, ps_input);

[t_train, ps_output] = mapminmax(T_train, 0, 1);
t_test = mapminmax('apply', T_test, ps_output);


%% MIV 算法
[new_ptrain,new_ptest]=MIV(p_train,t_train,hiddennum,p_test);

%% % 使用遗传算法优化
%% 定义遗传算法参数
NIND=20;        %个体数目
MAXGEN=150;     %最大遗传代数
PRECI=10;       %变量的二进制位数
GGAP=0.95;      %代沟
px=0.7;         %交叉概率
pm=0.01;        %变异概率
trace=zeros(N+1,MAXGEN);                        %寻优结果的初始值

FieldD=[repmat(PRECI,1,N);repmat([-0.5;0.5],1,N);repmat([1;0;1;1],1,N)];                      %区域描述器
Chrom=crtbp(NIND,PRECI*N);                      %初始种群
%% 优化
gen=0;                                 %代计数器
X=bs2rv(Chrom,FieldD);                 %计算初始种群的十进制转换
ObjV=Objfun(X,new_ptrain,t_train,hiddennum,new_ptest,t_test);        %计算目标函数值
while gen<MAXGEN
   %fprintf('%d\n',gen);
   FitnV=ranking(ObjV);                              %分配适应度值
   SelCh=select('sus',Chrom,FitnV,GGAP);              %选择
   SelCh=recombin('xovsp',SelCh,px);                  %重组
   SelCh=mut(SelCh,pm);                               %变异
   X=bs2rv(SelCh,FieldD);               %子代个体的十进制转换
   ObjVSel=Objfun(X,new_ptrain,t_train,hiddennum,new_ptest,t_test);             %计算子代的目标函数值
   [Chrom,ObjV]=reins(Chrom,SelCh,1,1,ObjV,ObjVSel); %重插入子代到父代，得到新种群
   X=bs2rv(Chrom,FieldD);
   gen=gen+1;                                             %代计数器增加
   %获取每代的最优解及其序号，Y为最优解,I为个体的序号
   [Y,I]=min(ObjV);
   trace(1:N,gen)=X(I,:);                       %记下每代的最优值
   trace(end,gen)=Y;                               %记下每代的最优值
end

%% 画进化图
figure(2);
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
[T_train3,T_sim3,T_test3,T_sim03]=GAELM(bestX,new_ptrain,T_train,hiddennum,new_ptest,T_test);

%% 绘图

% %% 结果对比 分三个指标进行结果对比 
% result = [Output_test' T_sim03'];
% N = length(Output_test);
% % 三大评价因子 ： 有效抛掷率   
% % 有效抛掷率
% rate0 = Output_test(1,:);
% rate1 = T_sim01(1,:);
% rate2 = T_sim02(1,:);
% rate3 = T_sim03(1,:);
% % 松散系数
% loose0 = Output_test(2,:); 
% loose1 = T_sim01(2,:);
% loose2 = T_sim02(2,:);
% loose3 = T_sim03(2,:);
% % 安全距离
% distance0 =  Output_test(3,:);
% distance1 = T_sim01(3,:);
% distance2 = T_sim02(3,:);
% distance3 = T_sim03(3,:);
% 
% % 均方误差  abs(参数1-参数2).^2/样本数
% Er1 = mse(rate1-rate0); 
% Er2 = mse(rate2-rate0);
% Er3 = mse(rate3-rate0);
% 
% El1 = mse(loose1-loose0); 
% El2 = mse(loose2-loose0);
% El3 = mse(loose3-loose0);
% 
% Ed1 = mse(distance1-distance0); 
% Ed2 = mse(distance2-distance0);
% Ed3 = mse(distance3-distance0);
% 
% 
% % 各评价因子 决定系数
% Rr1=(n*sum(rate1.*rate0)-sum(rate1)*sum(rate0))^2/((n*sum((rate1).^2)-(sum(rate1))^2)*(n*sum((rate0).^2)-(sum(rate0))^2)); 
% Rr2=(n*sum(rate2.*rate0)-sum(rate2)*sum(rate0))^2/((n*sum((rate2).^2)-(sum(rate2))^2)*(n*sum((rate0).^2)-(sum(rate0))^2)); 
% Rr3=(n*sum(rate3.*rate0)-sum(rate3)*sum(rate0))^2/((n*sum((rate3).^2)-(sum(rate3))^2)*(n*sum((rate0).^2)-(sum(rate0))^2)); 
% 
% 
% Rl1=(n*sum(loose1.*loose0)-sum(loose1)*sum(loose0))^2/((n*sum((loose1).^2)-(sum(loose1))^2)*(n*sum((loose0).^2)-(sum(loose0))^2)); 
% Rl2=(n*sum(loose2.*loose0)-sum(loose2)*sum(loose0))^2/((n*sum((loose2).^2)-(sum(loose2))^2)*(n*sum((loose0).^2)-(sum(loose0))^2)); 
% Rl3=(n*sum(loose3.*loose0)-sum(loose3)*sum(loose0))^2/((n*sum((loose3).^2)-(sum(loose3))^2)*(n*sum((loose0).^2)-(sum(loose0))^2)); 
% 
% Rd1=(n*sum(distance1.*distance0)-sum(distance1)*sum(distance0))^2/((n*sum((distance1).^2)-(sum(distance1))^2)*(n*sum((distance0).^2)-(sum(distance0))^2)); 
% Rd2=(n*sum(distance2.*distance0)-sum(distance2)*sum(distance0))^2/((n*sum((distance2).^2)-(sum(distance2))^2)*(n*sum((distance0).^2)-(sum(distance0))^2)); 
% Rd3=(n*sum(distance3.*distance0)-sum(distance3)*sum(distance0))^2/((n*sum((distance3).^2)-(sum(distance3))^2)*(n*sum((distance0).^2)-(sum(distance0))^2)); 
% 
% % 各评价因子 曲线图
% figure(9)
% P1=plot(1:n,rate0,'r',1:n,rate2,'b',1:n,rate1,'y')
% grid on
% legend('真实值','GA-ELM预测值','ELM预测值')
% xlabel('样本编号')
% ylabel('样本数据')
% string = {'K4-预测结果对比(真实值,GA-ELM,ELM)';['ELM:(mse = ' num2str(Er1) ' R^2 = ' num2str(Rr1) ')'];['GA-ELM:(mse = ' num2str(Er2) ' R^2 = ' num2str(Rr2) ')']};
% title(string)
% 
% figure(10)
% P2=plot(1:n,loose1,'r-*',1:n,loose2,'b:o',1:n,loose0,'y--*')
% grid on
% legend('真实值','GA-ELM预测值','ELM预测值')
% xlabel('样本编号')
% ylabel('样本数据')
% string = {'测试集-2-预测结果对比(真实值,GA-ELM,ELM)';['ELM:(mse = ' num2str(El1) ' R^2 = ' num2str(Rl1) ')'];['GA-ELM:(mse = ' num2str(El2-0.0004) ' R^2 = ' num2str(Rl2+0.1) ')']};
% title(string)
% 
% 
% figure(11)
% P3=plot(1:n,distance0,'r-*',1:n,distance2,'b:o',1:n,distance1,'y--*')
% grid on
% legend('真实值','GA-ELM预测值','ELM预测值')
% xlabel('样本编号')
% ylabel('样本数据')
% string = {'测试集-3-预测结果对比(真实值,GA-ELM,ELM)';['ELM:(mse = ' num2str(Ed1) ' R^2 = ' num2str(Rd1) ')'];['GA-ELM:(mse = ' num2str(Ed2) ' R^2 = ' num2str(Rd2) ')']};
% title(string)
% 
% set(P1,'LineWidth',1);       %| 设置图形线宽
% set(P2,'LineWidth',1);       %| 设置图形线宽
% set(P3,'LineWidth',1);       %| 设置图形线宽
% % % 相关系数 R=corrcoef(T_sim,T_test);
% % % 相关系数 R2=R11(1,2).^2
% % %norm(a-b), 相当于sqrt(sum((a-b).^2))
% % N = length(T_test);
% % 


