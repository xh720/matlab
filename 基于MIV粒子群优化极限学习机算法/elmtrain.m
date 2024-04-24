function [LW, TF] = elmtrain(p_train, t_train, TF, TYPE, IW, B)

%%  Description
% Input
% P   - Input Matrix of Training Set  (R*Q)  ѵ����������
% T   - Output Matrix of Training Set (S*Q) ѵ���������
% TF  - Transfer Function: ���ݺ�����ת������
%       'sig' for Sigmoidal function (default) S�ͺ���
%       'sin' for Sine function ���Һ���
%       'hardlim' for Hardlim function Ӳ�����ʹ��ݺ���
% TYPE - Regression (0,default) or Classification (1)
% IW   - Input Weight Matrix (N*R) ����Ȩֵ
% B    - Bias Matrix  (N*1) ƫ��
% Output
% LW  - Layer Weight Matrix (N*S)

if size(p_train, 2) ~= size(t_train, 2)  
    error('ELM:Arguments', 'The columns of P and T must be same.');
end

%%  ת�����ģʽ
if TYPE  == 1
    t_train  = ind2vec(t_train);
end

%%  ��ʼ��Ȩ��
[~, Q] = size(t_train);
BiasMatrix = repmat(B, 1, Q);

%%  �������
tempH = IW * p_train + BiasMatrix;

%%  ѡ�񼤻��
switch TF
    case 'sig'
        H = 1 ./ (1 + exp(-tempH));
    case 'sin'
        H = sin(tempH);
    case 'hardlim'
        H = hardlim(tempH);
end

%%  α�����Ȩ��
LW = pinv(H') * t_train';