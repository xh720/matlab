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

%%  ������ʼ������
hiddennum = 50;                   % ���ز�ά�� BP��������ͨ��С������ڵ�����������ڵ���
indim     = size(p_train, 1);     % �����ά�� 7
outdim    = size(t_train, 1);     % �����ά�� 1

%%  ����Ⱥ��������
c1      = 1.45;         % ѧϰ����
c2      = 1.64;         % ѧϰ����
iternum =   30;         % ��Ⱥ��ģ
Pmax    =    1;         % ��Ⱥ���ֵ
Pmin    =   -1;         % ��Ⱥ��Сֵ
Vmax    =  0.5;         % ����ٶ�
Vmin    = -0.5;         % ��С�ٶ�
wmax    =  0.8;         % ����������ʼֵ
wmin    =  0.1;         % ����������ֵֹ
itmax   =  100;         % ��Ⱥ���´���  
minerr  = 1e-6;         % �����ֵ

%%  ��Ⱥ��ʼ��

% ������Ӧ��ֵ�Ĵ洢�ռ�
MinFit  = [];
BestFit = [];

% ��ʼ��Ȩ�ع������ӳ�ʼ��
for iter = 1: itmax
    W(iter) = wmax - ((wmax - wmin) / itmax) * iter; 
end

% ��ʼ�����Ż�������Ŀ  �����ڵ�����1�ٳ������ز�ڵ���
D_nodes = (indim + 1) * hiddennum;  %������������Ȩ�غ�ƫ�õ�����

% ��ʼ����Ⱥ���ٶ�
Pop = 2 * rand(iternum, D_nodes, 1) - 1; % ��
V   = 2 * rand(iternum, D_nodes, 1) - 1;

%%  ��ʼ������Ⱥ������
fitness = fitcal(Pop, indim, hiddennum, p_train, t_train);  % ��ʼ����Ӧ��ֵ
fvrec(:, 1, 1) = fitness(:, 1, 1);                          % �洢��Ӧ��ֵ
[C, I] = min(fitness(:, 1, 1));                             % �õ������Ӧ��ֵ
MinFit  = [MinFit , C];                                     % �洢���������Ӧ��ֵ
BestFit = [BestFit, C];                                     % �洢ȫ�������Ӧ��ֵ
L(:, 1, 1) = fitness(:, 1, 1);                              % 
B(1, 1, 1) = C;                                             % ��ǰ��Ӧ��ֵ
gbest(1, :, 1) = Pop(I, :, 1);                              % ���������Ⱥ

for i = 1 : iternum
    zbest(i, :, 1) = Pop(1, :, 1);                          % ȫ�������Ⱥ
    pbest(i, :, 1) = Pop(i, :, 1);                          % ���������Ⱥ
end

%%  �õ���ǰ�ٶ�ֵ
V(:, :, 2) = W(1) * V(:, :, 1) + c1 * rand * (pbest(:, :, 1) - Pop(:, :, 1)) ...
              + c2 * rand * (zbest(:, :, 1) - Pop(:, :, 1));

%%  �����ٶȱ߽�
for ni = 1 : iternum
    for di = 1 : D_nodes

        if V(ni, di, 2) > Vmax
            % �����ϱ߽�
            V(ni, di, 2) = Vmax;
        elseif V(ni, di, 2) < Vmin
            % �����±߽�
            V(ni, di, 2) = Vmin;
        end

    end
end

%%  ��Ⱥ����
Pop(:, :, 2) = Pop(:, :, 1) + V(:, :, 2);

%%  ������Ⱥ�߽�
for ni = 1 : iternum
    for di = 1 : D_nodes

        if Pop(ni, di, 2) > Pmax
            % ��Ⱥ�Ͻ�
            Pop(ni, di, 2) = Pmax;
        elseif V(ni, di, 2) < -Pmin
            % ��Ⱥ�½�
            Pop(ni, di, 2) = -Pmin;
        end

    end
end

%%  �����Ż�
for j = 2 : itmax

    % �õ���Ӧ��ֵ
    fitness = fitcal(Pop, indim, hiddennum, p_train, t_train); 

    % ��������
    fvrec(:, 1, j) = fitness(:, 1, j);    %����ǰ��Ӧ��ֵ�洢��fvrec����ĵ�һ����
    [C, I] = min(fitness(:, 1, j));
    MinFit = [MinFit, C];                   % �洢���������Ӧ��ֵ
    BestFit = [BestFit, min(MinFit)];       % �洢���ȫ����Ӧ��ֵ
    L(:, 1, j) = fitness(:, 1, j);          % �ۼ���Ӧ��ֵ
    B(1, 1, j) = C;                         % �洢��С�����ֵ
    gbest(1, :, j) = Pop(I, :, j);          % �õ�ȫ�������Ⱥ
    [C, I] = min(B(1, 1, :));               % ��С������
    
    % �õ�ȫ�������Ⱥ
    if B(1, 1, j) <= C
        gbest(1, :, j) = gbest(1, :, j);
    else
        gbest(1, :, j) = gbest(1, :, I);
    end
    
    % ������Ӧ��ֵ��С����ֵ������ѭ��
    if (C <= minerr)
        break
    end

    % ������������������������������ѭ��
    if (j >= itmax)
        break
    end

    % �洢ȫ�������Ⱥ
    for p = 1 : iternum
        zbest(p, :, j) = gbest(1, :, j);
    end

    % �õ����������Ⱥ
    for i = 1 : iternum
        [C, I] = min(L(i, 1, :));
        if L(i, 1, j) <= C
            pbest(i, :, j) = Pop(i, :, j);
        else
            pbest(i, :, j) = Pop(i, :, I);
        end
    end

    % �����ٶ�
    V(:, :, j + 1) = W(j) * V(:, :, j) + c1 * rand * (pbest(:, :, j) - ...
        Pop(:, :, j)) + c2 * rand*(zbest(:, :, j) - Pop(:, :, j));
    
    % �ٶȱ߽�����
    for ni = 1 : iternum
        for di = 1 : D_nodes

            if V(ni, di, j + 1) > Vmax
                % �����ٶ�����
               V(ni, di, j + 1) = Vmax;
            elseif V(ni, di, j + 1) < Vmin
                % �����ٶ�����
               V(ni, di, j + 1) = Vmin;
            end

        end
    end

    % ������Ⱥ
    Pop(:, :, j + 1) = Pop(:, :, j) + V(:, :, j + 1);
    
    % ��Ⱥ�߽�����
    for ni = 1 : iternum
        for di = 1 : D_nodes

            if Pop(ni, di, j + 1) > Pmax
               % ������Ⱥ�߽�����
               Pop(ni, di, j + 1) = Pmax;
            elseif V(ni, di, j + 1) < Pmin
               % ������Ⱥ�߽�����
               Pop(ni, di, j + 1) = Pmin;
            end

        end
    end 

end

%%  ��ȡ����Ȩ��
for t = 1 : hiddennum
    IWbest(t, :) = gbest(1, ((t - 1) * indim + 1) : t * indim, j);
end
IBbest = gbest(1, end - hiddennum + 1 : end, j)';

%%  ����ģ��
[LW, TF] = elmtrain(p_train, t_train, 'sig', 0, IWbest, IBbest);

%%  ����Ԥ��
t_sim1 = elmpredict(p_train, IWbest, IBbest, LW, TF);
t_sim2 = elmpredict(p_test , IWbest, IBbest, LW, TF);

%%  ���ݷ���һ��
T_sim1 = mapminmax('reverse', t_sim1, ps_output);
T_sim2 = mapminmax('reverse', t_sim2, ps_output);

%%  ���������
RMSE1 = sqrt(sum((T_sim1 - T_train).^2) ./ M);
RMSE2 = sqrt(sum((T_sim2 - T_test ).^2) ./ N);

%%  ��ͼ
figure
plot(1: M, T_train, 'r-*', 1: M, T_sim1, 'b-o', 'LineWidth', 1)
legend('��ʵֵ', 'Ԥ��ֵ')
xlabel('Ԥ������')
ylabel('Ԥ����')
string = {'ѵ����Ԥ�����Ա�'; ['RMSE=' num2str(RMSE1)]};
title(string)
xlim([1, M])
grid

figure
plot(1: N, T_test, 'r-*', 1: N, T_sim2, 'b-o', 'LineWidth', 1)
legend('��ʵֵ', 'Ԥ��ֵ')
xlabel('Ԥ������')
ylabel('Ԥ����')
string = {'���Լ�Ԥ�����Ա�'; ['RMSE=' num2str(RMSE2)]};
title(string)
xlim([1, N])
grid

%%  ������ߵ���ͼ
figure;
plot(1: length(BestFit), BestFit, 'LineWidth', 1.5);
xlabel('����Ⱥ��������');
ylabel('��Ӧ��ֵ');
xlim([1, length(BestFit)])
string = {'ģ�͵������仯'};
title(string)
grid on

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
error1=T_sim1-T_train;
error2=T_sim2-T_test;
MAPE1=mean(abs(error1./T_train));
MAPE2=mean(abs(error2./T_test));

disp(['ѵ����Ԥ��׼ȷ��Ϊ��',num2str(100-MAPE1*100),'%'])
disp(['���Լ�Ԥ��׼ȷ��Ϊ��',num2str(100-MAPE2*100),'%'])

ms_pso = abs(T_sim2-T_test)./T_test;
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

pre2_pso=T_sim2;
r2_pso=R2;
mae_pso=mae2;
rmse_pso=RMSE2;

save PSO.mat pre2_pso r2_pso mae_pso rmse_pso ms_pso T_test