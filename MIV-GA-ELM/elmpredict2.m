function Y2 = elmpredict2(new_ptest, IW, B, LW, TF, TYPE)

%%  计算隐层输出
Q = size(new_ptest, 2);
BiasMatrix = repmat(B, 1, Q);
tempH = IW * new_ptest + BiasMatrix;

%%  选择激活函数
switch TF
    case 'sig'
        H = 1 ./ (1 + exp(-tempH));
    case 'hardlim'
        H = hardlim(tempH);
end

%%  计算输出
Y2 = (H' * LW)';

%%  转化分类模式
if TYPE == 1
    temp_Y = zeros(size(Y2));
    for i = 1:size(Y2, 2)
        [~, index] = max(Y2(:, i));
        temp_Y(index, i) = 1;
    end
    Y2 = vec2ind(temp_Y); 
end