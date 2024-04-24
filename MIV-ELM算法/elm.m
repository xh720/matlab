%%  清空环境变量
warning off             % 关闭报警信息
close all               % 关闭开启的图窗
clear                   % 清空变量
clc                     % 清空命令行
%rng('default')
%%  导入数据
res = xlsread('F:\大四\数据集.xlsx');

%%  数据分析
num_size = 0.8;                              % 训练集占数据集比例
outdim = 1;                                  % 最后一列为输出
num_samples = size(res, 1);                  % 样本个数
res = res(randperm(num_samples), :);         % 打乱数据集（不希望打乱时，注释该行）
num_train_s = round(num_size * num_samples); % 训练集样本个数
f_ = size(res, 2) - outdim;                  % 输入特征维度

%%  划分训练集和测试集
P_train = res(1: num_train_s, 1: f_)';   %1到 num_train_s行和1到f_列    训练集输入
T_train = res(1: num_train_s, f_ + 1: end)'; %为什么要转至？因为使用了mapminmax函数归一化，函数的要求。 训练集输出
M = size(P_train, 2);    %2代表P_train的列数，正常来说是行数，这里转至了就是列数

P_test = res(num_train_s + 1: end, 1: f_)';
T_test = res(num_train_s + 1: end, f_ + 1: end)';
N = size(P_test, 2);

%%  数据归一化
[p_train, ps_input] = mapminmax(P_train, 0, 1); %将训练集输入划分到0到1之间得到归一化之后的训练集输入以及一个归一化准则ps_input，
p_test = mapminmax('apply', P_test, ps_input); %将测试集按照训练集的归一化准则进行归一化得到归一化的测试集输入。

[t_train, ps_output] = mapminmax(T_train, 0, 1); %将训练集输出划分到0到1之间得到归一化之后的训练集输出，以及一个归一化准则，
t_test = mapminmax('apply', T_test, ps_output); %将测试集输出按照训练集输出的归一化准则去进行归一化，得到归一化后的测试集输出。

%%  创建模型
num_hiddens = 50;        % 隐藏层节点个数
activate_model = 'sig';  % 激活函数
[IW, B, LW, TF, TYPE] = elmtrain(p_train, t_train, num_hiddens, activate_model, 0);
% 设置完参数后，传入elmtrain这个函数中，
%0代表回归，1表示分类。IW输入到隐藏层的权重，B偏置，LW隐藏层到输出层的权重，TF是激活函数的类型，TYPE是回归还是分类的一个类型。
%% 通过初始样本数据集X建立新的训练样本
%将初始样本数据集X中每个自变量的指标值增加10%得到新的训练样本Q。
%将初始样本数据集X中第i个指标的值增加10%,且其他自变量的值保持不变所构成的样本数据集用Qi表示。
p=p_train;
%varNum = size( inputData,1);    % 变量个数
% ======用训练好的网络计算各个变量的MIV值=====================
p=p';
[m,n]=size(p);
yy_temp=p;
% p_increase为增加10%的矩阵 p_decrease为减少10%的矩阵
for i=1:n
    p=yy_temp;
    pX=p(:,i);
    pa=pX*1.1;
    p(:,i)=pa;
    aa=['p_increase'  int2str(i) '=p;'];
    eval(aa);
end
for i=1:n
    p=yy_temp;
    pX=p(:,i);
    pa=pX*0.9;
    p(:,i)=pa;
    aa=['p_decrease' int2str(i) '=p;'];
    eval(aa);
end

%%  仿真测试
% 转置后sim

for i=1:n
    eval(['p_increase',num2str(i),'=transpose(p_increase',num2str(i),');'])
end
for i=1:n
    eval(['p_decrease',num2str(i),'=transpose(p_decrease',num2str(i),');'])
end

for i=1:n
    bb=['result_in',num2str(i), '= elmpredict(p_increase',num2str(i),',IW, B, LW, TF, TYPE);']; %加
    eval(bb)
end
for i=1:n
    cc=['result_de',num2str(i), '= elmpredict(p_decrease',num2str(i),',IW, B, LW, TF, TYPE);']; %加
    eval(cc)
end

for i=1:n
    IV= ['result_in',num2str(i), '-result_de',num2str(i)];
    eval(['MIV_',num2str(i) ,'=mean(',IV,')*(1e7)',';']) ;
    eval(['MIVX=', 'MIV_',num2str(i),';']);
    MIV(i,:)=abs(MIVX);
end
[MB,iranked] = sort(MIV,'descend');

%% 数据可视化分析
%%
%-------------------------------------------------------------------------------------
figure()
barh(MIV(iranked),'g');
xlabel('Variable Importance','FontSize',12,'Interpreter','latex');
ylabel('Variable Rank','FontSize',12,'Interpreter','latex');
title('特征变量重要度','fontsize',12,'FontName','华文宋体')
hold on
barh(MIV(iranked(1:3)),'y');
hold on
barh(MIV(iranked(1:3)),'r');
grid on 
xt = get(gca,'XTick');    
xt_spacing=unique(diff(xt));
xt_spacing=xt_spacing(1);    
yt = get(gca,'YTick');    
% 条形标注
for ii=1:length(MIV)
    text(...
        max([0 MIV(iranked(ii))+0.02*max(MIV)]),ii,...
        ['P ' num2str(iranked(ii))],'Interpreter','latex','FontSize',12);
end
set(gca,'FontSize',12)
set(gca,'YTick',yt);
set(gca,'TickDir','out');
set(gca, 'ydir', 'reverse' )
set(gca,'LineWidth',2);
grid on;
%drawnow
%[outputt,predict_label]  = ELM(iranked); %调用BP.m脚本，进行诊断
%% 
%-------------------------------------------------------------------------------------
iranked = iranked';
p_train=p_train';
p_test=p_test';
if sum(length(iranked)==[1,2,3])==3
    new_data = p_train;
else
    ir = iranked(1:2);  %取重要度较高的前4个特征作为神经网络的输入
    new_data = p_train(:,ir);
    new_data1=p_test(:,ir);
end
rng('default')

%% 使用筛选后的变量去建立模型

p_train=new_data';
p_test=new_data1';
[IW, B, LW, TF, TYPE] = elmtrain(p_train, t_train, num_hiddens, activate_model, 0);
%%  筛选后仿真测试
t_sim1 = elmpredict(p_train, IW, B, LW, TF, TYPE);  %得到预测值
t_sim2 = elmpredict(p_test , IW, B, LW, TF, TYPE);

%%  数据反归一化
T_sim1 = mapminmax('reverse', t_sim1, ps_output); 
T_sim2 = mapminmax('reverse', t_sim2, ps_output);

%%  均方根误差
error1 = sqrt(sum((T_sim1 - T_train).^2) ./ M); %训练集误差
error2 = sqrt(sum((T_sim2 - T_test ).^2) ./ N); %测试集误差

%%  绘图
figure
plot(1: M, T_train, 'r-*', 1: M, T_sim1, 'b-o', 'LineWidth', 1)
legend('真实值', '预测值')
xlabel('预测样本')
ylabel('预测结果')
string = {'训练集预测结果对比'; ['RMSE=' num2str(error1)]};
title(string)
xlim([1, M])
grid

figure
plot(1: N, T_test, 'r-*', 1: N, T_sim2, 'b-o', 'LineWidth', 1)
legend('真实值', '预测值')
xlabel('预测样本')
ylabel('预测结果')
string = {'测试集预测结果对比'; ['RMSE=' num2str(error2)]};
title(string)
xlim([1, N])
grid

%%  相关指标计算
%  R2
R1 = 1 - norm(T_train - T_sim1)^2 / norm(T_train - mean(T_train))^2;  %mean求均值
R2 = 1 - norm(T_test  - T_sim2)^2 / norm(T_test  - mean(T_test ))^2;

disp(['训练集数据的R2为：', num2str(R1)])
disp(['测试集数据的R2为：', num2str(R2)])

%  MAE
mae1 = sum(abs(T_sim1 - T_train)) ./ M ;  %M为样本数
mae2 = sum(abs(T_sim2 - T_test )) ./ N ;

disp(['训练集数据的MAE为：', num2str(mae1)])
disp(['测试集数据的MAE为：', num2str(mae2)])

%  MBE
mbe1 = sum(T_sim1 - T_train) ./ M ;
mbe2 = sum(T_sim2 - T_test ) ./ N ;

disp(['训练集数据的MBE为：', num2str(mbe1)])
disp(['测试集数据的MBE为：', num2str(mbe2)])

ms_miv = abs(T_sim2-T_test)./T_test;
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

r2_miv=R2;
pre2_miv=T_sim2;
mae_miv=mae2;
rmse_miv=error2;

save MIVELM.mat pre2_miv r2_miv mae_miv rmse_miv ms_miv T_test