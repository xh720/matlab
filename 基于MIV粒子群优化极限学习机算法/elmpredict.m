function Y = elmpredict(P, IW, B, LW, TF)

%%  偏置初始化
Q = size(P, 2);
BiasMatrix = repmat(B, 1, Q);

%%  隐藏层输出
tempH = IW * P + BiasMatrix;

%%  选择激活函数
switch TF
    case 'sig'
        H = 1 ./ (1 + exp(-tempH));
    case 'sin'
        H = sin(tempH);
    case 'hardlim'
        H = hardlim(tempH);
end

%%  得到输出值
Y = (H' * LW)';
