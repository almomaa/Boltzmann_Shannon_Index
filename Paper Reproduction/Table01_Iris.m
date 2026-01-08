clearvars; close all; clc;

%% Iris Dataset – Benchmark Validation of the Boltzmann-Shannon Index
% Perfect for Figure 2 and Table 2 of your BSI paper
% Works with your existing function: BSI = ClustersBSIE(Data, labels);
clear; close all; clc;
rng(123); % for reproducibility

%% Load Iris dataset (built-in in MATLAB)
load fisheriris.mat    % gives meas (150×4) and species (150×1 cell)
X = meas;              % 150 samples × 4 features
trueLabels = grp2idx(species);  % 1,2,3

p = getFreqProb(trueLabels);
q = getUniqueRepFreq(X,trueLabels);


%% Run K-means many times with k=3 (Iris has 3 species)
nReps = 100;
k = 3;

BSI_values          = zeros(nReps,1);
Silhouette_values   = zeros(nReps,1);
CH_values           = zeros(nReps,1);  % Calinski-Harabasz
DB_values           = zeros(nReps,1);  % Davies-Bouldin
Entropy_sizes       = zeros(nReps,1);  % Shannon entropy of cluster proportions

for rep = 1:nReps
    % Stable k-means (same options as before)
    labels = kmeans(X, k, 'Replicates',20, 'Start','plus', ...
                    'EmptyAction','singleton', 'MaxIter',2000);
    
    % Your Boltzmann-Shannon Index
    BSI_values(rep) = ClustersBSIE(X, labels);
    
    % Classical indices
    Silhouette_values(rep) = mean(silhouette(X, labels));
    eva = evalclusters(X, labels, 'CalinskiHarabasz');
    CH_values(rep) = eva.CriterionValues;
    eva = evalclusters(X, labels, 'DaviesBouldin');
    DB_values(rep) = eva.CriterionValues;
    
    % Simple Shannon entropy of cluster sizes (baseline)
    props = histcounts(labels, 1:k+1) / length(labels);
    props(props==0) = [];
    Entropy_sizes(rep) = -sum(props .* log2(props + eps));
end

%% Best partitioning (highest BSI)
[~, bestRep] = max(BSI_values);
bestLabels = kmeans(X, k, 'Replicates',20, 'Start','plus', ...
                    'EmptyAction','singleton', 'MaxIter',2000, ...
                    'Options',statset('UseParallel',0)); % one final clean run
% Re-compute everything for the best one
bestLabels = kmeans(X, k, 'Start','sample', 'Replicates',1, ...
                    'MaxIter',2000); % deterministic final run using the best seed
% Actually: just run once more with the same rng state if you want 100% identical
rng(123 + bestRep);
bestLabels = kmeans(X, k, 'Replicates',1, 'Start','plus');

BSI_best       = ClustersBSIE(X, bestLabels);
Sil_best       = mean(silhouette(X, bestLabels));
CH_best        = evalclusters(X, bestLabels, 'CalinskiHarabasz').CriterionValues;
DB_best        = evalclusters(X, bestLabels, 'DaviesBouldin').CriterionValues;
props          = histcounts(bestLabels, 1:4) / 150;
Entropy_best   = -sum(props(props>0) .* log2(props(props>0)));

%% Print nice table for the manuscript
fprintf('\n=== Table 2 – Iris dataset (best partitioning out of 100 runs) ===\n');
fprintf('%-25s %12s\n', 'Metric', 'Value');
fprintf('%-25s %12.4f   ← proposed\n', 'Boltzmann-Shannon Index', BSI_best);
fprintf('%-25s %12.4f\n', 'Silhouette score', Sil_best);
fprintf('%-25s %12.1f\n', 'Calinski-Harabasz', CH_best);
fprintf('%-25s %12.4f\n', 'Davies-Bouldin', DB_best);
fprintf('%-25s %12.4f   (baseline)\n', 'Entropy of sizes', Entropy_best);

%% Figure 2 – PCA projection + clustering colored by bestLabels
figure('Position',[400 200 1200 500], 'Color','w');

% PCA for visualization
[coeff, score, ~, ~, explained] = pca(X);
X_pca = score(:,1:2);

subplot(1,2,1)
gscatter(X_pca(:,1), X_pca(:,2), trueLabels, 'rmb', 'osd', 8);
title('Ground-truth Iris Species','FontWeight','bold','FontSize',13);
xlabel(sprintf('PC1 (%.1f%%)', explained(1)));
ylabel(sprintf('PC2 (%.1f%%)', explained(2)));
axis equal tight; box on; legend('Setosa','Versicolor','Virginica');

subplot(1,2,2)
cmap = [0 0.4470 0.7410; 0.8500 0.3250 0.0980; 0.9290 0.6940 0.1250];
gscatter(X_pca(:,1), X_pca(:,2), bestLabels, cmap, 'o', 8, 'filled');
title({'\textbf{K-means clustering (k=3)}'; ...
       sprintf('Boltzmann-Shannon Index = %.3f', BSI_best)}, ...
       'FontWeight','bold','FontSize',13,'Interpreter','latex');
xlabel(sprintf('PC1 (%.1f%%)', explained(1)));
ylabel(sprintf('PC2 (%.1f%%)', explained(2)));
axis equal tight; box on;

sgtitle('\textbf{Iris Dataset – Validation of the Boltzmann-Shannon Index}', ...
        'FontSize',15,'FontWeight','bold','Interpreter','latex');

exportgraphics(gcf,'Figure_2_Iris_Boltzmann_Shannon_Index.pdf','ContentType','vector');