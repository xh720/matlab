function  [T_train,T_sim1,T_test,T_sim01]=ELM(P_train,T_train,P_test,T_test,hiddennumber)
%% 训练&测试BP网络
%% 输入
% x：一个个体的初始权值和阈值
% P：训练样本输入
% T：训练样本输出
% hiddennum：隐含层神经元数
% P_test:测试样本输入
% T_test:测试样本期望输出
%% 输出
% err：预测样本的预测误差的范数

% Input_train = res(1: num_train_s, 1: x)';     %训练输入
% Output_train = res(1: num_train_s, x + 1: end)';  %训练输出
% M = size(Input_train, 2);
% 
% Input_test = res(num_train_s + 1: end, 1: x)';
% Output_test = res(num_train_s + 1: end, x + 1: end)';
% N = size(Input_test, 2);

%%  数据归一化
[p_train, ps_input] = mapminmax(P_train, 0, 1);
p_test = mapminmax('apply', P_test, ps_input);

[t_train, ps_output] = mapminmax(T_train, 0, 1);
%output_test = mapminmax('apply', Output_test, ps_output);

%%  创建模型
activate_model = 'sig';  % 激活函数
[IW, B, LW, TF, TYPE] = elmtrain(p_train, t_train, hiddennumber, activate_model, 0);

%% %网络训练数据输出
train_sim1 = elmpredict(p_train, IW, B, LW, TF, TYPE);  %训练集输出
%% 网络测试输出
test_sim2 = elmpredict(p_test , IW, B, LW, TF, TYPE);  %测试集输出

%网络训练输出反归一化
Train_sim1 = mapminmax('reverse', train_sim1, ps_output); 
%网络测试输出反归一化
Test_sim2 = mapminmax('reverse', test_sim2, ps_output);

%% %误差输出
disp('【未优化前训练结果】')
%y=norm(Test_sim2-output_test);
%% 评价指标
m=length(P_train);
n=length(P_test);

%% 转化一下变量
T_sim1=Train_sim1;
T_sim01=Test_sim2;
% T_train=T_train;
% T_test=T_test;

%% %%决定系数
% R1 = 1 - (sum((T_sim1 - T_train).^2)/ sum((T_sim1 - mean(T_train)).^2));
% R2 = 1 - (sum((T_sim2 - T_test).^2) / sum((T_sim2 - mean(T_test)).^2));
R1 = 1 - norm(T_train - T_sim1)^2 / norm(T_train - mean(T_train))^2;  %mean求均值
R2 = 1 - norm(T_test  - T_sim01)^2 / norm(T_test  - mean(T_test ))^2;

%% 平均绝对误差MAE
MAE1=sum(abs(T_sim1 - T_train)) ./ m ;
MAE2=sum(abs(T_sim01 - T_test)) ./ n ;
% MAE1=mean(abs(T_sim1 - T_train)) ./ m ;
% MAE2=mean(abs(T_sim2 - T_test)) ./ n ;

%% 平均偏差MBE
mbe1 = sum(T_sim1 - T_train) ./ m ;
mbe2 = sum(T_sim01 - T_test ) ./ n ;


%% %%均方根误差 RMSE
error1 = sqrt(sum((T_sim1 - T_train).^2)./m);
error2 = sqrt(sum((T_sim01 - T_test).^2)./n);


%% %%均方误差MSE
mse1 = sum((T_sim1 - T_train).^2)./m;
mse2 = sum((T_sim01 - T_test).^2)./n;

%%  %% 剩余预测残差RPD 
% %可以使用"std"函数计算标准偏差，使用"mean"函数计算残差的平均值，然后将这两个值相除得到RPD。
% SE1=std(T_sim1-T_train);
% RPD1=std(T_train)/SE1;
% 
% SE2=std(T_sim2-T_test);
% RPD2=std(T_test)/SE2;
% 
%  %% 平均绝对百分比误差MAPE
MAPE1=mean(abs((T_train -T_sim1)./T_train));
MAPE2=mean(abs((T_test -T_sim01)./T_test));

%% %%打印出评价指标
disp( ' -----------------------训练集误差计算----------------------- ')
disp( 'ELM训练集的评价结果如下所示: ')
disp(['决定系数R^2为:      ' ,num2str(R1)])
disp(['平均绝对误差MAE为:   ' ,num2str(MAE1)])
disp(['平均偏差MBE为:      ' ,num2str(mbe1)])
disp(['均方误差MSE为:      ' ,num2str(mse1)])
disp(['均方根误差RMSE为:    ',num2str( error1)])
% disp(['剩余预测残差RPD为:     ' ,num2str(RPD1)])
disp(['平均绝对百分比误差MAPE为:     ' ,num2str(MAPE1)])
disp(['训练集预测准确率为：',num2str(100-MAPE1*100),'%'])

disp( ' -----------------------测试集误差计算----------------------- ')
disp( 'ELM测试集的评价结果如下所示:  ')
disp(['决定系数R^2为:      ', num2str(R2)])
disp(['平均绝对误差MAE为:    ', num2str(MAE2)])
disp(['平均偏差MBE为:      ' ,num2str(mbe2)])
disp(['均方误差MSE为:      ', num2str(mse2)])
disp(['均方根误差RMSE为:     ', num2str(error2)])
% disp(['剩余预测残差RPD为:    ', num2str(RPD2)])
disp(['平均绝对百分比误差MAPE为:     ', num2str(MAPE2)])
disp(['测试集预测准确率为：',num2str(100-MAPE2*100),'%'])

%%  绘图
figure(1)
plot(1: m, T_train, 'r-*', 1: m, T_sim1, 'b-o', 'LineWidth', 1)
legend('真实值', '预测值')
xlabel('预测样本')
ylabel('预测结果')
string = {'训练集预测结果对比'; ['(R^2=' num2str(R1) ' RMSE=' num2str(error1) ')']};
title(string)
xlim([1, m])
grid on

figure(2)
plot(1: n, T_test, 'r-*', 1: n, T_sim01, 'b-o', 'LineWidth', 1)
legend('真实值', '预测值')
xlabel('预测样本')
ylabel('预测结果')
string = {'测试集预测结果对比'; ['(R^2=' num2str(R2) ' RMSE=' num2str(error2) ')']};

title(string)
xlim([1, n])
grid on


end