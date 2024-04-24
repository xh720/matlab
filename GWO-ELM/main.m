clc;
clear all;
close all
addpath pathA
tic
rng('default')
%%  导入数据
res = xlsread('F:\大四\数据集.xlsx');
%%  数据分析
num_size = 0.8;                              % 训练集占数据集比例l
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

%%  初始隐层神经元个数
hiddennum=40;
inputnum=size(p_train,1);       % 输入层神经元个数
outputnum=size(t_train,1);      % 输出层神经元个数
w1num=inputnum*hiddennum; % 输入层到隐层的权值个数
w2num=outputnum*hiddennum;% 隐层到输出层的权值个数
dim=w1num+hiddennum+w2num+outputnum; %待优化的变量的个数

%%  参数设置
T=250;  %%迭代次数
pop=30;  %%种群数量
ub=1;  %%上限
lb=0;  %%下限
fobj=@(X)Objfun(X,p_train,t_train,hiddennum,p_test,t_test); %适应度函数
[Best_F,Best_P,BestCost]=GWO(pop,T,lb,ub,dim,fobj);  %%算法

%% 重新训练
w1=Best_P(1:w1num);   %初始输入层到隐层的权值
w1 = reshape(w1,hiddennum,inputnum);
B1=Best_P(w1num+1:w1num+hiddennum);  %初始隐层阈值
B1=reshape(B1,hiddennum,1); 

%% ELM 训练/预测
[LW,TF,TYPE] = elmtrain(p_train,t_train,hiddennum,'sig',0,w1,B1);
t_sim1 = elmpredict(p_train,w1,B1,LW,TF,TYPE);
t_sim2 = elmpredict(p_test,w1,B1,LW,TF,TYPE);

%%  数据反归一化
T_sim1 = mapminmax('reverse', t_sim1, ps_output);
T_sim2 = mapminmax('reverse', t_sim2, ps_output);

%%  均方根误差
error1 = sqrt(sum((T_sim1 - T_train).^2) ./ M);
error2 = sqrt(sum((T_sim2 - T_test ).^2) ./ N);
toc

%决定系数
R1 = 1 - norm(T_train - T_sim1)^2 / norm(T_train - mean(T_train))^2;
R2 = 1 - norm(T_test -  T_sim2)^2 / norm(T_test -  mean(T_test ))^2;

%% 平均偏差MBE
mbe1 = sum(T_sim1 - T_train) ./ M ;
mbe2 = sum(T_sim2 - T_test ) ./ N ;

%%
%均方误差 MSE
mse1 = sum((T_sim1 - T_train).^2)./M;
mse2 = sum((T_sim2 - T_test).^2)./N;
%%
%RPD 剩余预测残差
SE1=std(T_sim1-T_train);
RPD1=std(T_train)/SE1;

SE=std(T_sim2-T_test);
RPD2=std(T_test)/SE;
%% 平均绝对误差MAE
MAE1 = mean(abs(T_train - T_sim1));
MAE2 = mean(abs(T_test - T_sim2));
%% 平均绝对百分比误差MAPE
MAPE1 = mean(abs((T_train - T_sim1)./T_train));
MAPE2 = mean(abs((T_test - T_sim2)./T_test));

disp(['训练集预测准确率为：',num2str(100-MAPE1*100),'%'])
disp(['测试集预测准确率为：',num2str(100-MAPE2*100),'%'])


%%  适应度曲线
figure
plot(1 : length(BestCost), BestCost,'r-', 'LineWidth', 1.5);
title('GWO-ELM', 'FontSize', 10);
xlabel('迭代次数', 'FontSize', 10);
ylabel('适应度值', 'FontSize', 10);
grid off

%训练集绘图
figure
plot(1: M, T_train,  'r-*','linewidth',1);
hold on 
plot(1: M, T_sim1,  'b-o','linewidth',1);
legend('真实值','GWO-ELM预测值')
xlabel('训练集样本')
ylabel('预测结果')
string = {'训练集预测结果对比'; ['(R^2=' num2str(R1) ' RMSE=' num2str(error1) ')']};
title(string)
xlim([1, M])
grid off

%测试集绘图
figure
plot(1: N, T_test,  'r-*','linewidth',1);
hold on 
plot(1: N, T_sim2,  'b-o','linewidth',1);
legend('真实值','GWO-ELM预测值')
xlabel('测试集样本')
ylabel('预测结果')
string = {'测试集预测结果对比'; ['(R^2=' num2str(R2) ' RMSE=' num2str(error2) ')']};
title(string)
xlim([1, N])
grid off

%% 测试集结果
figure;
plotregression(T_test,T_sim2,['回归图']);
figure;
ploterrhist(T_test-T_sim2,['误差直方图']);

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
%%  相关指标计算
disp(['训练集数据误差：'])
[mae_train,mse_train,rmse_train,mape_train,error_train,errorPercent_train,R_train]=calc_error(T_train,T_sim1); %
disp(['测试集数据误差：'])
[mae_test,mse_test,rmse_test,mape_test,error_test,errorPercent_test,R_test]=calc_error(T_test,T_sim2); %

pre2_gwo=T_sim2;
r2_gwo=R2;
mae_gwo=MAE2;
rmse_gwo=error2;

save GWO.mat pre2_gwo r2_gwo mae_gwo rmse_gwo T_test


