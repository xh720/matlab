function Obj=Objfun2(X,new_ptrain,output_train,hiddennumber,new_ptest,output_test)
%% 用来分别求解种群中各个个体的目标值
%% 输入
% X：所有个体的初始权值和阈值
% P：训练样本输入
% T：训练样本输出
% hiddennum：隐含层神经元数
% P_test:测试样本输入
% T_test:测试样本期望输出
%% 输出
% Obj：所有个体的预测样本的预测误差的范数

[M,N]=size(X);
Obj=zeros(M,1);
% input_train = P ;
% output_train = T ;
for i=1:M
    Obj(i)=ELMfun2(X(i,:),new_ptrain,output_train,hiddennumber,new_ptest,output_test);
end