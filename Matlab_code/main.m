clear; clc;

% ---- CONFIG ----
csvFile = 'results_log_models_temp.csv';             
outDir  = 'figures_checking';
metric  = 'F2';
panelFontSize = 11;
lineWidth = 1.8;

addpath(fullfile(pwd, 'code_for_figures'));
if ~exist(outDir, 'dir'); mkdir(outDir); end

% ---- Run only for specific data_ids ----

fh_all = plot_horizon_performance_all_dataset( ...
    csvFile, ...
    'metric',      metric, ...
    'outDir',      outDir, ...
    'titleStr',    sprintf('%s vs Horizon — All datasets', metric), ...
    'lineWidth',   lineWidth, ...
    'baseFontSize',panelFontSize);

fprintf('Saved ALL-DATASETS FIG: %s\n', fh_all.UserData.outputPathFIG);

% If you want it to pop open for editing right away:
try
    h = openfig(fh_all.UserData.outputPathFIG, 'new', 'visible');
    figure(h);  % bring to front
catch ME
    warning('Could not open FIG automatically: %s', ME.message);
end