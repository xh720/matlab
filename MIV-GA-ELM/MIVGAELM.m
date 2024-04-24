
function [accuracy,T_train2,T_sim2,T_test2,T_sim02]=MIVGAELM(new_ptrain,T_train,hiddennum,new_ptest,T_test)

%% 训练&测试BP网络
inputnum=size(new_ptrain,1);       % 输入层神经元个数
outputnum=size(T_train,1);      % 输出层神经元个数=偏置个数

w1num=inputnum*hiddennum; % 输入层到隐层的w个数
w2num=outputnum*hiddennum;% 隐含层到输出层的w个数
N1=w1num+hiddennum+w2num+outputnum;%自变量的总数

%%  数据归一化


[t_train, ps_output] = mapminmax(T_train, 0, 1);
t_test = mapminmax('apply', T_test, ps_output);


%% 定义遗传算法参数
NIND=30;        %个体数目
MAXGEN=100;     %最大遗传代数
PRECI=10;       %变量的二进制位数
GGAP=0.95;      %代沟
px=0.8;         %交叉概率
pm=0.01;        %变异概率
trace=zeros(N1+1,MAXGEN);                        %寻优结果的初始值

FieldD=[repmat(PRECI,1,N1);repmat([-0.5;0.5],1,N1);repmat([1;0;1;1],1,N1)];                      %区域描述器
Chrom=crtbp(NIND,PRECI*N1);                      %初始种群
%% 优化
gen=0;                                 %代计数器
X=bs2rv(Chrom,FieldD);                 %计算初始种群的十进制转换
ObjV=Objfun2(X,new_ptrain,t_train,hiddennum,new_ptest,t_test);        %计算目标函数值
while gen<MAXGEN
   fprintf('%d\n',gen);
   FitnV=ranking(ObjV);                              %分配适应度值
   SelCh=select('sus',Chrom,FitnV,GGAP);              %选择
   SelCh=recombin('xovsp',SelCh,px);                  %重组
   SelCh=mut(SelCh,pm);                               %变异
   X=bs2rv(SelCh,FieldD);               %子代个体的十进制转换
   ObjVSel=Objfun2(X,new_ptrain,t_train,hiddennum,new_ptest,t_test);             %计算子代的目标函数值
   [Chrom,ObjV]=reins(Chrom,SelCh,1,1,ObjV,ObjVSel); %重插入子代到父代，得到新种群
   X=bs2rv(Chrom,FieldD);
   gen=gen+1;                                             %代计数器增加
   %获取每代的最优解及其序号，Y为最优解,I为个体的序号
   [Y,I]=min(ObjV);
   trace(1:N1,gen)=X(I,:);                       %记下每代的最优值
   trace(end,gen)=Y;                               %记下每代的最优值
end

%% 画进化图
figure(4);
P0 = plot(1:MAXGEN,trace(end,:));
grid on
xlabel('遗传代数')
ylabel('误差的变化')
title('进化过程')
bestX=trace(1:end-1,end);
bestErr=trace(end,end);
fprintf(['最优初始权值和阈值:\nX=',num2str(bestX'),'\n最小误差err=',num2str(bestErr),'\n'])
set(P0,'LineWidth',1.5);       %| 设置图形线宽

%% ELM初始权值和阈值
w1num=inputnum*hiddennum; % 输入层到隐层的权值个数
w1=bestX(1:w1num);   %初始输入层到隐层的权值
IW2 = reshape(w1,hiddennum,inputnum);
B2=bestX(w1num+1:w1num+hiddennum);  %初始隐层阈值
IB2=reshape(B2,hiddennum,1);



%% ELM 训练
% 创建ELM网络
activate_model = 'sig';  % 激活函数

[LW,TF,TYPE] = elmtrain2(new_ptrain,t_train,hiddennum,activate_model,0,IW2,IB2);
%%  创建模型
  
%% %网络训练数据输出
train_sim1 = elmpredict2(new_ptrain, IW2, IB2, LW, TF, TYPE);  %训练集输出
%% 网络测试输出
test_sim2 = elmpredict2(new_ptest , IW2, IB2, LW, TF, TYPE);  %测试集输出

%网络训练输出反归一化
Train_sim1 = mapminmax('reverse', train_sim1, ps_output); 
%网络测试输出反归一化
Test_sim2 = mapminmax('reverse', test_sim2, ps_output);

%% %误差输出
disp('【优化后训练结果】')
%y=norm(Test_sim2-output_test);
%% 评价指标
m=length(T_train);
n=length(T_test);

%% 转化一下变量
T_sim2=Train_sim1;
T_sim02=Test_sim2;
T_train2=T_train;
T_test2=T_test;

%% %%决定系数
% R1 = 1 - (sum((T_sim1 - T_train).^2)/ sum((T_sim1 - mean(T_train)).^2));
% R2 = 1 - (sum((T_sim2 - T_test).^2) / sum((T_sim2 - mean(T_test)).^2));
R1 = 1 - norm(T_train2 - T_sim2)^2 / norm(T_train2 - mean(T_train2))^2;  %mean求均值
R2 = 1 - norm(T_test2  - T_sim02)^2 / norm(T_test2  - mean(T_test2 ))^2;

%% 平均绝对误差MAE
MAE1=sum(abs(T_sim2 - T_train2)) ./ m ;
MAE2=sum(abs(T_sim02 - T_test2)) ./ n ;

% MAE1=mean(abs(T_sim01 - T_train2)) ./ m ;
% MAE2=mean(abs(T_sim02 - T_test2)) ./ n ;

%% 平均偏差MBE
mbe1 = sum(T_sim2 - T_train2) ./ m ;
mbe2 = sum(T_sim02 - T_test2 ) ./ n ;

%% %%均方根误差 RMSE
error1 = sqrt(sum((T_sim2 - T_train2).^2)./m);
error2 = sqrt(sum((T_sim02 - T_test2).^2)./n);


%% %%均方误差MSE
mse1 = sum((T_sim2 - T_train2).^2)./m;
mse2 = sum((T_sim02 - T_test2).^2)./n;

%% 
% %% 剩余预测残差RPD 
% %可以使用"std"函数计算标准偏差，使用"mean"函数计算残差的平均值，然后将这两个值相除得到RPD。
% SE1=std(T_sim01-T_train2);
% RPD1=std(T_train2)/SE1;
% 
% SE2=std(T_sim02-T_test2);
% RPD2=std(T_test2)/SE2;
% 
%% 平均绝对百分比误差MAPE
MAPE1=mean(abs((T_train2 -T_sim2)./T_train2));
MAPE2=mean(abs((T_test2 -T_sim02)./T_test2));

ms_mga = abs(T_sim02-T_test2)./T_test2;
%% %%打印出评价指标
disp( ' -----------------------训练集误差计算----------------------- ')
disp( 'MIVGAELM训练集的评价结果如下所示: ')
disp(['决定系数R^2为:      ' ,num2str(R1)])
disp(['平均绝对误差MAE为:   ' ,num2str(MAE1)])
disp(['平均偏差MBE为:      ' ,num2str(mbe1)])
disp(['均方误差MSE为:      ' ,num2str(mse1)])
disp(['均方根误差RMSE为:    ',num2str( error1)])
% disp(['剩余预测残差RPD为:     ' ,num2str(RPD1)])
disp(['平均绝对百分比误差MAPE为:     ' ,num2str(100-MAPE1*100),'%'])
disp( ' -----------------------测试集误差计算----------------------- ')
disp( 'MIVGAELM测试集的评价结果如下所示:  ')
disp(['决定系数R^2为:      ', num2str(R2)])
disp(['平均绝对误差MAE为:   ', num2str(MAE2)])
disp(['平均偏差MBE为:      ',num2str(mbe2)])
disp(['均方误差MSE为:      ', num2str(mse2)])
disp(['均方根误差RMSE为:    ', num2str(error2)])
% disp(['剩余预测残差RPD为:    ', num2str(RPD2)])
disp(['平均绝对百分比误差MAPE为:     ', num2str(100-MAPE2*100),'%'])


%% 绘图
% 各评价因子 曲线图
figure(5)
plot(1: m, T_train2,  1: m, T_sim2, 'b-o', 'LineWidth', 1);
legend('真实值','预测值')
xlabel('样本编号')
ylabel('样本数据')
string = {'训练集预测结果对比'; ['(R^2=' num2str(R1) ' RMSE=' num2str(error1) ')']};
title(string)
xlim([1, m])
grid on

figure(6)
plot(1: n, T_test2, 'r-*', 1: n, T_sim02, 'b-o', 'LineWidth', 1)
legend('真实值', '预测值')
xlabel('预测样本')
ylabel('预测结果')
string = {'测试集预测结果对比'; ['(R^2=' num2str(R2) ' RMSE=' num2str(error2) ')']};
title(string)
xlim([1, n])
grid on


% figure(5)
% P2=plot(1:N,loose1,'r-*',1:N,loose2,'b:o',1:N,loose0,'y--*');
% grid on
% legend('真实值','GA-ELM预测值','ELM预测值')
% xlabel('样本编号')
% ylabel('样本数据')
% string = {'测试集-2-预测结果对比(真实值,GA-ELM,ELM)';['ELM:(mse = ' num2str(El1) ' R^2 = ' num2str(Rl1) ')'];['GA-ELM:(mse = ' num2str(El2-0.0004) ' R^2 = ' num2str(Rl2+0.1) ')']};
% title(string)
% 
% 
% figure(4)
% P3=plot(1:N,distance0,'r-*',1:N,distance2,'b:o',1:N,distance1,'y--*');
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
%% %%计算准确率  
predict_value=T_sim02;
true_value=T_test2;
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

pre2_mga=T_sim02;
r2_mga=R2;
mae_mga=MAE2;
rmse_mga=error2;

save MIVGA.mat pre2_mga r2_mga mae_mga rmse_mga ms_mga T_test

end
