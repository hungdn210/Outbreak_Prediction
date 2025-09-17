clear; clc;

% ---- CONFIG ----
csvFile = 'results_log_models.csv';             
outDir  = 'figures_checking';
metric  = 'F2';
panelFontSize = 11;
lineWidth = 1.8;

addpath(fullfile(pwd, 'code_for_figures'));
if ~exist(outDir, 'dir'); mkdir(outDir); end

% ---- Run only for specific data_ids ----
targets = ["France_level_2_final"];

for i = 1:numel(targets)
    target_id = targets(i);

    fh = plot_horizon_performance_by_country( ...
        csvFile, ...
        'metric', metric, ...
        'data_id', target_id, ...   % NEW
        'outDir', outDir, ...
        'titleStr', sprintf('%s vs Horizon — %s', metric, target_id), ...
        'lineWidth', lineWidth, ...
        'baseFontSize', panelFontSize ...
    );

    fprintf('Saved FIG: %s\n', fh.UserData.outputPathFIG);
end
