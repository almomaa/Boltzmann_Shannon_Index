clearvars; close all; clc;

%% Synthetic 2D Gaussian Mixtures – FINAL BULLETPROOF VERSION
% For Figure 1 and Table 1 of your Boltzmann-Shannon Index paper
clear; close all; clc;
rng(42); % perfect reproducibility

%% Safe k-means (never returns empty clusters → no NaNs in ClustersBSIE)
safe_kmeans = @(X,k) kmeans(X,k, ...
    'Distance','sqeuclidean', ...
    'Replicates',100, ...
    'MaxIter',2000, ...
    'Start','plus', ...
    'EmptyAction','singleton', ...
    'OnlinePhase','off');

%% 1. Balanced & well-separated
muA    = [2 2; 6 6; 4 0];
SigmaA = cat(3, 0.40*eye(2), 0.50*eye(2), 0.45*eye(2));
X_A = [];
for i = 1:3
    X_A = [X_A; mvnrnd(muA(i,:), SigmaA(:,:,i), 400)];
end
labels_A = safe_kmeans(X_A, 3);
BSI_A = ClustersBSIE(X_A, labels_A);

%% 2. Moderately imbalanced
muB    = [2 2; 6 6; 4 0];
SigmaB = cat(3, 0.20*eye(2), 0.80*eye(2), 1.40*eye(2));
X_B = [mvnrnd(muB(1,:), SigmaB(:,:,1), 400); ...
       mvnrnd(muB(2,:), SigmaB(:,:,2), 100); ...
       mvnrnd(muB(3,:), SigmaB(:,:,3), 100)];
labels_B = safe_kmeans(X_B, 3);
BSI_B = ClustersBSIE(X_B, labels_B);

%% 3. Poorly separated + highly imbalanced
muC    = [2 2; 6 6; 4 0];
SigmaC = cat(3, 0.05*eye(2), 0.80*eye(2), 3.40*eye(2));
X_C = [mvnrnd(muC(1,:), SigmaC(:,:,1), 1000); ...
       mvnrnd(muC(2,:), SigmaC(:,:,2), 200); ...
       mvnrnd(muC(3,:), SigmaC(:,:,3), 20)];
% Tiny jitter – guarantees no perfectly duplicate points (extra safety)
X_C = X_C + 1e-12*randn(size(X_C));

labels_C = safe_kmeans(X_C, 3);
BSI_C = ClustersBSIE(X_C, labels_C);


fprintf('=== Boltzmann-Shannon Index (higher = better & more balanced) ===\n');
fprintf('Balanced & well-separated)      → BSI = %.4f\n', BSI_A);
fprintf('Moderately imbalanced            → BSI = %.4f\n', BSI_B);
fprintf('Poor separation + high imbalance → BSI = %.4f\n\n', BSI_C);

%% Beautiful publication-ready Figure 1
figure('Position',[300 200 1400 480], 'Color','w');
cmap = [0 0.4470 0.7410; 0.8500 0.3250 0.0980; 0.9290 0.6940 0.1250];

subplot(3,1,1)
gscatter(X_A(:,1), X_A(:,2), labels_A, cmap, '.', 10,'off');
title({'\textbf{Balanced}'; sprintf('BSI = %.3f', BSI_A)}, ...
      'FontSize',11, 'Interpreter','latex');
axis equal tight; box on;
set(gca,"FontSize",12)
axis([0 10 -2 8])

subplot(3,1,2)
gscatter(X_B(:,1), X_B(:,2), labels_B, cmap, '.', 10,'off');
title({'\textbf{Moderately Imbalanced}'; sprintf('BSI = %.3f', BSI_B)}, ...
      'FontSize',11, 'Interpreter','latex');
axis equal tight; box on;
set(gca,"FontSize",12)
axis([0 10 -2 8])

subplot(3,1,3)
gscatter(X_C(:,1), X_C(:,2), labels_C, cmap, '.', 10,'off');
title({'\textbf{Imbalance}'; sprintf('BSI = %.3f', BSI_C)}, ...
      'FontSize',11, 'Interpreter','latex');
axis equal tight; box on;
axis([0 10 -2 8])

% sgtitle('\textbf{The Boltzmann-Shannon Index on Synthetic 2D Gaussian Mixtures}', ...
%         'FontSize',15, 'Interpreter','latex');

% exportgraphics(gcf,'Figure_1_Boltzmann_Shannon_Index.pdf','ContentType','vector');


set(gca,"FontSize",12)
set(gcf, 'Units', 'Inches', 'Position', [0, 0, 3, 3], 'PaperUnits', 'Inches', 'PaperSize', [7.25, 9.125])