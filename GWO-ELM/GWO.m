
function [Alpha_score,Alpha_pos,Convergence_curve]=GWO(SearchAgents_no,Max_iteration,lb,ub,dim,fobj)
%SearchAgents_no ���ǵ���Ⱥ����

%%  �Ż��㷨��ʼ��
Alpha_pos = zeros(1, dim);  % ��ʼ��Alpha�ǵ�λ��
Alpha_score = inf;          % ��ʼ��Alpha�ǵ�Ŀ�꺯��ֵ,�������Ϊ-inf�Խ���������

Beta_pos = zeros(1, dim);   % ��ʼ��Beta�ǵ�λ��
Beta_score = inf;           % ��ʼ��Beta�ǵ�Ŀ�꺯��ֵ ,�������Ϊ-inf�Խ���������

Delta_pos = zeros(1, dim);  % ��ʼ��Delta�ǵ�λ��
Delta_score = inf;          % ��ʼ��Delta�ǵ�Ŀ�꺯��ֵ,�������Ϊ-inf�Խ���������

%%  ��ʼ��������Ⱥ��λ��
Positions = initialization(SearchAgents_no, dim, ub, lb);

%%  ���ڼ�¼��������
Convergence_curve = zeros(1, Max_iteration);
%%  ѭ��������
iter = 0;

%%  �Ż��㷨��ѭ��
while iter < Max_iteration           % �Ե�������ѭ��
    for i = 1 : size(Positions, 1)   % ����ÿ����

        % ���س��������ռ�߽��������Ⱥ
        % ������λ�ó����������ռ䣬��Ҫ���»ص������ռ�
        Flag4ub = Positions(i, :) > ub;
        Flag4lb = Positions(i, :) < lb;

        % ���ǵ�λ�������ֵ����Сֵ֮�䣬��λ�ò���Ҫ���������������ֵ����ص����ֵ�߽�
        % ��������Сֵ����ش���Сֵ�߽�
        Positions(i, :) = (Positions(i, :) .* (~(Flag4ub + Flag4lb))) + ub .* Flag4ub + lb .* Flag4lb;   

        % ������Ӧ�Ⱥ���ֵ
%         Positions(i, 2) = round(Positions(i, 2));
%         fitness = fical(Positions(i, :));
          fitness = fobj(Positions(i, :));
        % ���� Alpha, Beta, Delta
        if fitness < Alpha_score           % ���Ŀ�꺯��ֵС��Alpha�ǵ�Ŀ�꺯��ֵ
            Alpha_score = fitness;         % ��Alpha�ǵ�Ŀ�꺯��ֵ����Ϊ����Ŀ�꺯��ֵ
            Alpha_pos = Positions(i, :);   % ͬʱ��Alpha�ǵ�λ�ø���Ϊ����λ��
        end

        if fitness > Alpha_score && fitness < Beta_score   % ���Ŀ�꺯��ֵ������Alpha�Ǻ�Beta�ǵ�Ŀ�꺯��ֵ֮��
            Beta_score = fitness;                          % ��Beta�ǵ�Ŀ�꺯��ֵ����Ϊ����Ŀ�꺯��ֵ
            Beta_pos = Positions(i, :);                    % ͬʱ����Beta�ǵ�λ��
        end

        if fitness > Alpha_score && fitness > Beta_score && fitness < Delta_score  % ���Ŀ�꺯��ֵ������Beta�Ǻ�Delta�ǵ�Ŀ�꺯��ֵ֮��
            Delta_score = fitness;                                                 % ��Delta�ǵ�Ŀ�꺯��ֵ����Ϊ����Ŀ�꺯��ֵ
            Delta_pos = Positions(i, :);                                           % ͬʱ����Delta�ǵ�λ��
        end

    end

    % ����Ȩ�صݼ�
    wa = 2 - iter * ((2) / Max_iteration);    

    % ����������Ⱥ��λ��
    for i = 1 : size(Positions, 1)      % ����ÿ����
        for j = 1 : size(Positions, 2)  % ����ÿ��ά��

            % ��Χ���λ�ø���
            r1 = rand; % r1 is a random number in [0,1]
            r2 = rand; % r2 is a random number in [0,1]

            A1 = 2 * wa * r1 - wa;   % ����ϵ��A��Equation (3.3)
            C1 = 2 * r2;             % ����ϵ��C��Equation (3.4)

            % Alpha λ�ø���
            D_alpha = abs(C1 * Alpha_pos(j) - Positions(i, j));   % Equation (3.5)-part 1
            X1 = Alpha_pos(j) - A1 * D_alpha;                     % Equation (3.6)-part 1

            r1 = rand; % r1 is a random number in [0,1]
            r2 = rand; % r2 is a random number in [0,1]

            A2 = 2 * wa * r1 - wa;   % ����ϵ��A��Equation (3.3)
            C2 = 2 *r2;              % ����ϵ��C��Equation (3.4)

            % Beta λ�ø���
            D_beta = abs(C2 * Beta_pos(j) - Positions(i, j));    % Equation (3.5)-part 2
            X2 = Beta_pos(j) - A2 * D_beta;                      % Equation (3.6)-part 2       

            r1 = rand;  % r1 is a random number in [0,1]
            r2 = rand;  % r2 is a random number in [0,1]

            A3 = 2 *wa * r1 - wa;     % ����ϵ��A��Equation (3.3)
            C3 = 2 *r2;               % ����ϵ��C��Equation (3.4)

            % Delta λ�ø���
            D_delta = abs(C3 * Delta_pos(j) - Positions(i, j));   % Equation (3.5)-part 3
            X3 = Delta_pos(j) - A3 * D_delta;                     % Equation (3.5)-part 3

            % λ�ø���
            Positions(i, j) = (X1 + X2 + X3) / 3;                 % Equation (3.7)

        end
    end

    % ���µ�����
    iter = iter + 1;    
    Convergence_curve(iter) = Alpha_score;
    disp(['��',num2str(iter),'�ε���'])
    disp(['current iteration is: ',num2str(iter), ', best fitness is: ', num2str(Alpha_score)]);
end

%%  ��¼��Ѳ���
% best_lr = Alpha_pos(1, 1);
% best_hd = Alpha_pos(1, 2);
% best_l2 = Alpha_pos(1, 3);
end