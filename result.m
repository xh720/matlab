%% 运行算法
% MY_ELM_Reg
% MY_KELM_Reg
% MY_BP_Reg
% MY_PSOKELM_Reg
% MY_GWOKELM_Reg

clc,clear,close all
addpath 基于极限学习机的数据回归预测
addpath 基于遗传算法优化极限学习机
addpath 基于粒子群优化极限学习机的数据回归预测
addpath MIV-ELM算法
addpath MIV-GA-ELM
addpath 基于MIV粒子群优化极限学习机算法
%% 对比
load('ELM.mat')
load('GA.mat')
load('PSO.mat')
load('MIVELM.mat')
load('MIVGA.mat')
load('MIVPSO.mat')


% figure(3)
% plot(1:length(yy1),yy1,'Color',[243 162 97]/255,'LineWidth',3);hold on
% plot(1:length(yy2),yy2,'Color',[33 158 188]/255,'LineWidth',3);hold on
% 
% grid on
% legend('PSO','GWO','FontSize',12,'FontName','Times New Roman')
% xlabel('进化代数','FontSize',12)
% ylabel('最优个体适应度','FontSize',12)
% title('进化曲线对比','FontSize',14)

% 预测结果与实际值
figure
% plot(T_test(:),'Color',[38 70 83]/255,'LineWidth',1);
plot(T_test(:),'red-*','LineWidth',1);
hold on
plot(pre2_(:),'blue-o','LineWidth',1);
plot(pre2_ga(:),'g-o','LineWidth',1);
% plot(pre2_(:),'Color',[40 114 113]/255,'LineWidth',1);
% plot(pre2_ga(:),'Color',[138 176 125]/255,'LineWidth',1);
plot(pre2_pso(:),'Color',[233 196 107]/255,'LineWidth',1);
% plot(pre2_miv(:),'Color',[243 162 97]/255,'LineWidth',1);
% plot(pre2_mga(:),'Color',[33 158 188]/255,'LineWidth',1);
% plot(pre2_mpso(:),'Color',[70 200 168]/255,'LineWidth',1);
legend('实际值','ELM','GA-ELM','PSO-ELM','MIV-ELM','MIV-GA-ELM','MIV-PSO-ELM')
xlabel('测试样本编号')
ylabel('输出')
title('预测结果显示（测试集）')
grid on;


% % 预测结果与实际值
% figure
% plot(T_test(:),'Color',[38 70 83]/255,'LineWidth',2);
% hold on
% plot(pre2_(:),'Color',[40 114 113]/255,'LineWidth',2);
% plot(pre2_ga(:),'Color',[138 176 125]/255,'LineWidth',2);
% plot(pre2_pso(:),'Color',[233 196 107]/255,'LineWidth',2);
% plot(pre2_miv(:),'Color',[243 162 97]/255,'LineWidth',2);
% plot(pre2_mpso(:),'Color',[33 158 188]/255,'LineWidth',2);
% legend('实际值','ELM','GA-ELM','PSO-ELM','MIV-ELM','MIV-PSO-ELM')
% xlabel('测试样本编号')
% ylabel('输出')
% title('预测结果显示（测试集）')
% grid on;



% 预测相对误差
figure
plot(ms_(:),'Color',[40 114 113]/255,'LineWidth',2);hold on
plot(ms_ga(:),'Color',[138 176 125]/255,'LineWidth',2);hold on
plot(ms_pso(:),'Color',[233 196 107]/255,'LineWidth',2);hold on
plot(ms_miv(:),'Color',[243 162 97]/255,'LineWidth',2);
plot(ms_mga(:),'Color',[33 158 188]/255,'LineWidth',2);
plot(ms_mpso(:),'Color',[70 200 168]/255,'LineWidth',1);
legend('ELM','GA-ELM','PSO-ELM','MIV-ELM','MIV-GA-ELM','MIV-PSO-ELM')
xlabel('测试样本编号','FontSize',12);
ylabel('相对误差','FontSize',12);
title('测试集的相对误差')
grid on

figure
% 误差统计
AA = [ r2_,r2_ga r2_pso r2_miv r2_mga r2_mpso;...
    mae_,mae_ga mae_pso mae_miv mae_mga mae_mpso;...
    rmse_,rmse_ga rmse_pso rmse_miv rmse_mga rmse_mpso];
   
B= bar(AA);
xticklabels({'R2' ,' MAE',  'RMSE' })
legend('ELM','GA-ELM','PSO-ELM','MIV-ELM','MIV-GA-ELM','MIV-PSO-ELM','FontName','Times New Roman','FontSize',11)
B(1).FaceColor = [40 114 113]/255;
B(2).FaceColor = [138 176 125]/255;
B(3).FaceColor = [233 196 107]/255;
B(4).FaceColor = [243 162 97]/255;
B(5).FaceColor = [33 158 188]/255;
B(6).FaceColor = [70 200 168]/255;

title('预测算法误差对比')

