function err=ELMfun(x,P_train,T_train,hiddennum,P_test,T_test)
inputnum=size(P_train,1);       % �������Ԫ���� 
outputnum=size(T_train,1);      % �������Ԫ����
%% ELM��ʼȨֵ����ֵ
w1num=inputnum*hiddennum; %   ����㵽�����Ȩֵ����
w1=x(1:w1num);   %��ʼ����㵽�����Ȩֵ
w1 = reshape(w1,hiddennum,inputnum);
B1=x(w1num+1:w1num+hiddennum);  %��ʼ������ֵ
B1=reshape(B1,hiddennum,1);
%% ELM ѵ��
[LW,TF,TYPE] = elmtrain(P_train,T_train,hiddennum,'sig',0,w1,B1);
% ELM �������
T_sim = elmpredict(P_test,w1,B1,LW,TF,TYPE);
err=mean(power((T_sim-T_test),2));
end

 

