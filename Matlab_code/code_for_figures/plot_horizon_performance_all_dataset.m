function fh = plot_horizon_performance_all_dataset(csvFile, varargin)
% plot_horizon_performance_all_dataset
% One figure pooling ALL datasets: metric vs horizon, one line per MODEL.
%
% Example:
%   fh = plot_horizon_performance_all_dataset('results_log_main.csv', ...
%           'metric','F2','outDir','figures','titleStr','F2 vs Horizon — All datasets');

% ---- Parse args ----
p = inputParser;
addParameter(p, 'metric', 'F2', @(s)ischar(s) || isstring(s));
addParameter(p, 'outDir', 'figures', @(s)ischar(s) || isstring(s));
addParameter(p, 'titleStr', '', @(s)ischar(s) || isstring(s));
addParameter(p, 'lineWidth', 1.8, @isscalar);
addParameter(p, 'baseFontSize', 11, @isscalar);
parse(p, varargin{:});
opt = p.Results;

metric = char(opt.metric);

% ---- Read CSV ----
T = readtable(csvFile, 'TextType','string');

% Guards
need = ["data_id","model","month",metric];
missing = setdiff(need, string(T.Properties.VariableNames));
if ~isempty(missing)
    error('Missing required columns in CSV: %s', strjoin(missing, ', '));
end

% ---- Derive helpers ----
% Country (not used for filtering here, but useful if needed later)
T.country = extractBefore(T.data_id, "_");
T.country(T.country=="") = T.data_id(T.country==""); % fallback

% Horizon number from 'Month+1'...'Month+6'
mh = regexp(string(T.month),'Month\+(\d+)','tokens','once');
T.horizon = nan(height(T),1);
for i = 1:height(T)
    tok = mh{i};
    if ~isempty(tok), T.horizon(i) = str2double(tok{1}); end
end
T = T(~isnan(T.horizon), :);

% Ensure metric numeric
if ~isnumeric(T.(metric)), T.(metric) = str2double(string(T.(metric))); end
T = T(~isnan(T.(metric)), :);

% ---- Aggregate across ALL datasets: mean metric per (model, horizon) ----
G = groupsummary(T, {'model','horizon'}, 'mean', metric);
G.Properties.VariableNames{end} = 'MeanMetric'; % rename mean_metric column

% Get model list in a stable order
models = unique(G.model, 'stable');
N = numel(models);

% Build palette + marker sets for clarity
cols = tableau20_colors(min(N,20));
markers = {'o','s','^','v','d','>','<','p','h','x','+'};

% ---- Plot ----
fh = figure('Color','w','Units','normalized','Position',[0.15 0.15 0.65 0.55],'Visible','off');
ax = axes(fh); hold(ax,'on'); grid(ax,'on'); box(ax,'on');

% For each model, get mean metric per horizon and plot
for m = 1:N
    sub = G(G.model==models(m), :);
    % Ensure horizons 1..K exist (fill with NaN if some horizons missing)
    H = sub.horizon;
    Kmax = max(H);
    x = (1:Kmax).';
    y = nan(Kmax,1);
    [~, ia] = ismember(H, x);
    y(ia) = sub.MeanMetric;

    mk = markers{mod(m-1, numel(markers))+1};
    plot(x, y, '-', ...
        'Color', cols(m,:), ...
        'LineWidth', opt.lineWidth, ...
        'Marker', mk, 'MarkerFaceColor', cols(m,:), 'MarkerSize', 5, ...
        'DisplayName', char(models(m)));
end

xlabel(ax, 'Forecast Horizon (Month +k)', 'FontSize', opt.baseFontSize);
ylabel(ax, sprintf('Average %s (across all datasets)', metric), 'FontSize', opt.baseFontSize);

ttl = opt.titleStr;
if isempty(ttl), ttl = sprintf('%s vs Forecast Horizon — All datasets', metric); end
title(ax, ttl, 'FontWeight','bold', 'FontSize', opt.baseFontSize+2);

set(ax, 'FontSize', opt.baseFontSize, 'XLim',[0.8 6.2], 'XTick',1:6);
leg = legend(ax, 'Location','southoutside', 'NumColumns', 2); leg.Box = 'off';

% ---- Save FIG ----
if ~exist(opt.outDir, 'dir'), mkdir(opt.outDir); end
baseName = sprintf('performance_over_horizon_%s_all_datasets', lower(metric));
outFIG = fullfile(opt.outDir, baseName + ".fig");

% Save FIG (for editing in MATLAB)
savefig(fh, outFIG);

fh.UserData.outputPathFIG = outFIG;
fh.UserData.summary = G;

end

% -------------- Local helper: Tableau-20 palette --------------
function C = tableau20_colors(n)
% Return the first n colors from Tableau 20 (distinct, publication-friendly).
base = [ ...
     31 119 180; 255 127  14;  44 160  44; 214  39  40; 148 103 189; ...
    140  86  75; 227 119 194; 127 127 127; 188 189  34;  23 190 207; ...
    174 199 232; 255 187 120; 152 223 138; 255 152 150; 197 176 213; ...
    196 156 148; 247 182 210; 199 199 199; 219 219 141; 158 218 229 ...
] / 255;
n = min(n, size(base,1));
C = base(1:n,:);
end
