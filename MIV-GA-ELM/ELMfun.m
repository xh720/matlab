function err=ELMfun(x,Input_train,Output_train,hiddennumber,Input_test,Output_test)
%% 训练&测试网络
%% 输入
% x：一个个体的初始权值和阈值
% P：训练样本输入
% T：训练样本输出
% hiddennum：隐含层神经元数
% P_test:测试样本输入
% T_test:测试样本期望输出
%% 输出
% err：预测样本的预测误差的范数

%%  数据归一化
[input_train, ps_input] = mapminmax(Input_train, 0, 1);
input_test = mapminmax('apply', Input_test, ps_input);

[output_train, ps_output] = mapminmax(Output_train, 0, 1);
%output_test = mapminmax('apply', Output_test, ps_output);



inputnum=size(Input_train,1);       % 输入层神经元个数 
%outputnum=size(Output_train,1);      % 输出层神经元个数

% N = size(P_test,1); % 获取测试集的列数，即测试样本数


%% ELM初始权值和阈值
w1num=inputnum*hiddennumber; % 输入层到隐层的权值个数
w1=x(1:w1num);   %初始输入层到隐层的权值
w1 = reshape(w1,hiddennumber,inputnum);
B1=x(w1num+1:w1num+hiddennumber);  %初始隐层阈值
B1=reshape(B1,hiddennumber,1);

%% ELM 训练
% 创建ELM网络
activate_model = 'sig';  % 激活函数

[LW,TF,TYPE] = elmtrain(input_train,output_train,hiddennumber,activate_model,0,w1,B1);
%%  创建模型
  
% ELM仿真测试
t_sim = elmpredict(input_test,w1,B1,LW,TF,TYPE);
% 反归一化
T_sim = mapminmax('reverse',t_sim,ps_output);

err=norm(T_sim-Output_test);
end