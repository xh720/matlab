clc;
clear all;
close all
addpath pathA
tic
rng('default')
%%  ��������
res = xlsread('F:\����\���ݼ�.xlsx');
%%  ���ݷ���
num_size = 0.8;                              % ѵ����ռ���ݼ�����l
outdim = 1;                                  % ���һ��Ϊ���
num_samples = size(res, 1);                  % ��������
res = res(randperm(num_samples), :);         % �������ݼ�����ϣ������ʱ��ע�͸��У�
num_train_s = round(num_size * num_samples); % ѵ������������
f_ = size(res, 2) - outdim;                  % ��������ά��

%%  ����ѵ�����Ͳ��Լ�
P_train = res(1: num_train_s, 1: f_)';
T_train = res(1: num_train_s, f_ + 1: end)';
M = size(P_train, 2);

P_test = res(num_train_s + 1: end, 1: f_)';
T_test = res(num_train_s + 1: end, f_ + 1: end)';
N = size(P_test, 2);

%%  ���ݹ�һ��
[p_train, ps_input] = mapminmax(P_train, 0, 1);
p_test = mapminmax('apply', P_test, ps_input);

[t_train, ps_output] = mapminmax(T_train, 0, 1);
t_test = mapminmax('apply', T_test, ps_output);

%%  ��ʼ������Ԫ����
hiddennum=40;
inputnum=size(p_train,1);       % �������Ԫ����
outputnum=size(t_train,1);      % �������Ԫ����
w1num=inputnum*hiddennum; % ����㵽�����Ȩֵ����
w2num=outputnum*hiddennum;% ���㵽������Ȩֵ����
dim=w1num+hiddennum+w2num+outputnum; %���Ż��ı����ĸ���

%%  ��������
T=250;  %%��������
pop=30;  %%��Ⱥ����
ub=1;  %%����
lb=0;  %%����
fobj=@(X)Objfun(X,p_train,t_train,hiddennum,p_test,t_test); %��Ӧ�Ⱥ���
[Best_F,Best_P,BestCost]=GWO(pop,T,lb,ub,dim,fobj);  %%�㷨

%% ����ѵ��
w1=Best_P(1:w1num);   %��ʼ����㵽�����Ȩֵ
w1 = reshape(w1,hiddennum,inputnum);
B1=Best_P(w1num+1:w1num+hiddennum);  %��ʼ������ֵ
B1=reshape(B1,hiddennum,1); 

%% ELM ѵ��/Ԥ��
[LW,TF,TYPE] = elmtrain(p_train,t_train,hiddennum,'sig',0,w1,B1);
t_sim1 = elmpredict(p_train,w1,B1,LW,TF,TYPE);
t_sim2 = elmpredict(p_test,w1,B1,LW,TF,TYPE);

%%  ���ݷ���һ��
T_sim1 = mapminmax('reverse', t_sim1, ps_output);
T_sim2 = mapminmax('reverse', t_sim2, ps_output);

%%  ���������
error1 = sqrt(sum((T_sim1 - T_train).^2) ./ M);
error2 = sqrt(sum((T_sim2 - T_test ).^2) ./ N);
toc

%����ϵ��
R1 = 1 - norm(T_train - T_sim1)^2 / norm(T_train - mean(T_train))^2;
R2 = 1 - norm(T_test -  T_sim2)^2 / norm(T_test -  mean(T_test ))^2;

%% ƽ��ƫ��MBE
mbe1 = sum(T_sim1 - T_train) ./ M ;
mbe2 = sum(T_sim2 - T_test ) ./ N ;

%%
%������� MSE
mse1 = sum((T_sim1 - T_train).^2)./M;
mse2 = sum((T_sim2 - T_test).^2)./N;
%%
%RPD ʣ��Ԥ��в�
SE1=std(T_sim1-T_train);
RPD1=std(T_train)/SE1;

SE=std(T_sim2-T_test);
RPD2=std(T_test)/SE;
%% ƽ���������MAE
MAE1 = mean(abs(T_train - T_sim1));
MAE2 = mean(abs(T_test - T_sim2));
%% ƽ�����԰ٷֱ����MAPE
MAPE1 = mean(abs((T_train - T_sim1)./T_train));
MAPE2 = mean(abs((T_test - T_sim2)./T_test));

disp(['ѵ����Ԥ��׼ȷ��Ϊ��',num2str(100-MAPE1*100),'%'])
disp(['���Լ�Ԥ��׼ȷ��Ϊ��',num2str(100-MAPE2*100),'%'])


%%  ��Ӧ������
figure
plot(1 : length(BestCost), BestCost,'r-', 'LineWidth', 1.5);
title('GWO-ELM', 'FontSize', 10);
xlabel('��������', 'FontSize', 10);
ylabel('��Ӧ��ֵ', 'FontSize', 10);
grid off

%ѵ������ͼ
figure
plot(1: M, T_train,  'r-*','linewidth',1);
hold on 
plot(1: M, T_sim1,  'b-o','linewidth',1);
legend('��ʵֵ','GWO-ELMԤ��ֵ')
xlabel('ѵ��������')
ylabel('Ԥ����')
string = {'ѵ����Ԥ�����Ա�'; ['(R^2=' num2str(R1) ' RMSE=' num2str(error1) ')']};
title(string)
xlim([1, M])
grid off

%���Լ���ͼ
figure
plot(1: N, T_test,  'r-*','linewidth',1);
hold on 
plot(1: N, T_sim2,  'b-o','linewidth',1);
legend('��ʵֵ','GWO-ELMԤ��ֵ')
xlabel('���Լ�����')
ylabel('Ԥ����')
string = {'���Լ�Ԥ�����Ա�'; ['(R^2=' num2str(R2) ' RMSE=' num2str(error2) ')']};
title(string)
xlim([1, N])
grid off

%% ���Լ����
figure;
plotregression(T_test,T_sim2,['�ع�ͼ']);
figure;
ploterrhist(T_test-T_sim2,['���ֱ��ͼ']);

%% %%����׼ȷ��  
predict_value=T_sim2;
true_value=T_test;
correct = 0;  
total = length(predict_value);  
% ����ÿһ������  
for i = 1:total  
    if predict_value(i) <= true_value(i)+  true_value(i)*0.02 && predict_value(i) >= true_value(i)- true_value(i)*0.02
        correct = correct + 1;  
    end  
end   

accuracy = correct / total;

disp(['���׼ȷ��Ϊ��',num2str(accuracy)])
%%  ���ָ�����
disp(['ѵ����������'])
[mae_train,mse_train,rmse_train,mape_train,error_train,errorPercent_train,R_train]=calc_error(T_train,T_sim1); %
disp(['���Լ�������'])
[mae_test,mse_test,rmse_test,mape_test,error_test,errorPercent_test,R_test]=calc_error(T_test,T_sim2); %

pre2_gwo=T_sim2;
r2_gwo=R2;
mae_gwo=MAE2;
rmse_gwo=error2;

save GWO.mat pre2_gwo r2_gwo mae_gwo rmse_gwo T_test


