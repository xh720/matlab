%%  ��ջ�������
warning off             % �رձ�����Ϣ
close all               % �رտ�����ͼ��
clear                   % ��ձ���
clc                     % ���������

%%  ��������
res = xlsread('F:\����\���ݼ�.xlsx');

%%  ���ݷ���
num_size = 0.8;                              % ѵ����ռ���ݼ�����
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

%%  ����ģ��
num_hiddens = 50;        % ���ز�ڵ����
activate_model = 'sig';  % �����
[IW, B, LW, TF, TYPE] = elmtrain(p_train, t_train, num_hiddens, activate_model, 0);

%%  �������
t_sim1 = elmpredict(p_train, IW, B, LW, TF, TYPE);
t_sim2 = elmpredict(p_test , IW, B, LW, TF, TYPE);

%%  ���ݷ���һ��
T_sim1 = mapminmax('reverse', t_sim1, ps_output);
T_sim2 = mapminmax('reverse', t_sim2, ps_output);

%%  ���������
error1 = sqrt(sum((T_sim1 - T_train).^2) ./ M);
error2 = sqrt(sum((T_sim2 - T_test ).^2) ./ N);

%%  ��ͼ
figure
plot(1: M, T_train, 'r-*', 1: M, T_sim1, 'b-o', 'LineWidth', 1)
legend('��ʵֵ', 'Ԥ��ֵ')
xlabel('Ԥ������')
ylabel('Ԥ����')
string = {'ѵ����Ԥ�����Ա�'; ['RMSE=' num2str(error1)]};
title(string)
xlim([1, M])
grid

figure
plot(1: N, T_test, 'r-*', 1: N, T_sim2, 'b-o', 'LineWidth', 1)
legend('��ʵֵ', 'Ԥ��ֵ')
xlabel('Ԥ������')
ylabel('Ԥ����')
string = {'���Լ�Ԥ�����Ա�'; ['RMSE=' num2str(error2)]};
title(string)
xlim([1, N])
grid

%%  ���ָ�����
%  R2
R1 = 1 - norm(T_train - T_sim1)^2 / norm(T_train - mean(T_train))^2;
R2 = 1 - norm(T_test  - T_sim2)^2 / norm(T_test  - mean(T_test ))^2;

disp(['ѵ�������ݵ�R2Ϊ��', num2str(R1)])
disp(['���Լ����ݵ�R2Ϊ��', num2str(R2)])

%  MAE
mae1 = sum(abs(T_sim1 - T_train)) ./ M ;
mae2 = sum(abs(T_sim2 - T_test )) ./ N ;

disp(['ѵ�������ݵ�MAEΪ��', num2str(mae1)])
disp(['���Լ����ݵ�MAEΪ��', num2str(mae2)])

%  MBE
mbe1 = sum(T_sim1 - T_train) ./ M ;
mbe2 = sum(T_sim2 - T_test ) ./ N ;

disp(['ѵ�������ݵ�MBEΪ��', num2str(mbe1)])
disp(['���Լ����ݵ�MBEΪ��', num2str(mbe2)])

%MAPE
erro1=T_sim1-T_train;
erro2=T_sim2-T_test;
MAPE1=mean(abs(erro1./T_train));
MAPE2=mean(abs(erro2./T_test));

disp(['ѵ����Ԥ��׼ȷ��Ϊ��',num2str(100-MAPE1*100),'%'])
disp(['���Լ�Ԥ��׼ȷ��Ϊ��',num2str(100-MAPE2*100),'%'])

ms_ = abs(T_sim2-T_test)./T_test;
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



pre2_=T_sim2;
r2_=R2;
mae_=mae2;
rmse_=error2;

save ELM.mat pre2_ r2_ mae_ rmse_ ms_ T_test
