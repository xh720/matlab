function err=ELMfun(x,P_train,T_train,hiddennum,P_test,T_test)
inputnum=size(P_train,1);       % 输入层神经元个数 
outputnum=size(T_train,1);      % 输出层神经元个数
%% ELM初始权值和阈值
w1num=inputnum*hiddennum; %   输入层到隐层的权值个数
w1=x(1:w1num);   %初始输入层到隐层的权值
w1 = reshape(w1,hiddennum,inputnum);
B1=x(w1num+1:w1num+hiddennum);  %初始隐层阈值
B1=reshape(B1,hiddennum,1);
%% ELM 训练
[LW,TF,TYPE] = elmtrain(P_train,T_train,hiddennum,'sig',0,w1,B1);
% ELM 仿真测试
T_sim = elmpredict(P_test,w1,B1,LW,TF,TYPE);
err=mean(power((T_sim-T_test),2));
end

 

