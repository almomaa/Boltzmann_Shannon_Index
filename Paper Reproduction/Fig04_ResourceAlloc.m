clearvars; close all; clc;




out = randsrc(500000,1,[1 2 3; 0.95 0.049 0.001]);

% out = randsrc(500000,1,[1 2 3; 0.669 0.33 0.001]);


[~,~,ic] = unique(out);

h = accumarray(ic,1);

p = h/sum(h);



X = (randn(500000,2));
beta = 0;
beta = linspace(-1,1,60)';

for i=1:length(beta)
    r = 0.5*((1+beta(i))*p' + (1-beta(i))*fliplr(p'));

    Y = getNewSVD(X,ic,r);
    B(i) = ClustersBSIE(Y,ic);

    % scatter(Y(:,1),Y(:,2),8,ic,"filled")
    % colormap winter
    % pause(0.5)
    
end

% plot(beta,B')
figure('Position',[300 200 900 500],'Color','w');
plot(beta, B','-o','LineWidth',2,'MarkerFaceColor','b','MarkerSize',8);
grid on; box on;
xlabel('Fairness parameter $\beta$','FontSize',11,'Interpreter','latex');
ylabel('BSI($\beta$)','FontSize',11,'Interpreter','latex');
set(gca,'FontSize',20,'XTick',-1:0.25:1);
ylim([0 1]);


exportgraphics(gcf,'Figure_5_Resource_Allocation_3Communities.pdf','ContentType','vector');


