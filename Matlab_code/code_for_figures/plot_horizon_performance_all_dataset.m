function fh = plot_horizon_performance_all_dataset(csvFile, varargin)
% plot_horizon_performance_all_dataset
% One figure pooling ALL datasets: metric vs horizon, one line per MODEL.
%
% Example:
%   fh = plot_horizon_performance_all_dataset('results_log_main.csv', ...
%           'metric','F2','outDir','figures','topN',5, ...
%           'highlightModels',"Ensemble (mean weight) using 4 models", ...
%           'yLim',[0.65 0.95], 'titleStr','');

% ---- Parse args ----
p = inputParser;
addParameter(p, 'metric', 'F2', @(s)ischar(s) || isstring(s));
addParameter(p, 'outDir', 'figures', @(s)ischar(s) || isstring(s));
addParameter(p, 'titleStr', '', @(s)ischar(s) || isstring(s));
addParameter(p, 'lineWidth', 1.8, @isscalar);
addParameter(p, 'baseFontSize', 11, @isscalar);

% NEW
addParameter(p, 'topN', inf, @isscalar);  % keep only top-N models by overall mean
addParameter(p, 'highlightModels', strings(0), @(x)isstring(x)||iscellstr(x));
addParameter(p, 'yLim', [], @(v)isnumeric(v) && (isempty(v) || numel(v)==2));

parse(p, varargin{:});
opt = p.Results;

metric = char(opt.metric);
hiList = string(opt.highlightModels);   % normalize

% ---- Read CSV ----
T = readtable(csvFile, 'TextType','string');

% Guards
need = ["data_id","model","month",metric];
missing = setdiff(need, string(T.Properties.VariableNames));
if ~isempty(missing)
    error('Missing required columns in CSV: %s', strjoin(missing, ', '));
end

% ---- Derive helpers ----
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
G.Properties.VariableNames{end} = 'MeanMetric';

% ---- Rank models by overall mean across horizons ----
S = groupsummary(G, 'model', 'mean', 'MeanMetric');
S.Properties.VariableNames{end} = 'OverallMean';
S = sortrows(S, 'OverallMean', 'descend');

% Keep only top-N if requested
if isfinite(opt.topN)
    keep = S.model(1:min(opt.topN, height(S)));
    G = G(ismember(G.model, keep), :);
    S = S(ismember(S.model, keep), :);  % keep ranking table aligned
end

% Model order follows ranking (best first)
models = S.model;
N = numel(models);

% Build palette + marker sets for clarity
cols = tableau20_colors(min(N,20));
markers = {'o','s','^','v','d','>','<','p','h','x','+'};

% ---- Plot ----
fh = figure('Color','w','Units','normalized','Position',[0.15 0.15 0.65 0.55], 'Visible','on');
ax = axes(fh); hold(ax,'on'); grid(ax,'on'); box(ax,'on');

for m = 1:N
    sub = G(G.model==models(m), :);
    H = sub.horizon;
    Kmax = max(H);
    x = (1:Kmax).';
    y = nan(Kmax,1);
    [~, ia] = ismember(H, x);
    y(ia) = sub.MeanMetric;

    mk = markers{mod(m-1, numel(markers))+1};

    % Style: highlight vs de-emphasize
    isHi = ismember(string(models(m)), hiList);
    if ~isempty(hiList) && ~isHi
        lw = max(0.8, 0.8*opt.lineWidth);
        ls = '--';
    else
        lw = max(1.2, 1.6*opt.lineWidth); % make highlights thicker; also thicker when no highlight list
        ls = '-';
    end

    plot(x, y, ls, ...
        'Color', cols(m,:), ...
        'LineWidth', lw, ...
        'Marker', mk, 'MarkerFaceColor', cols(m,:), 'MarkerSize', 7, ...
        'DisplayName', char(models(m)));
end

xlabel(ax, 'Forecast Horizon (Month +k)', 'FontSize', opt.baseFontSize);
ylabel(ax, 'Average F2', 'FontSize', opt.baseFontSize);

ttl = opt.titleStr;
if isempty(ttl)
    ttl = 'All datasets';
end
title(ax, ttl, 'FontWeight','bold', 'FontSize', opt.baseFontSize+2);

set(ax, 'FontSize', opt.baseFontSize, 'XLim', [0.8 6.2], 'XTick', 1:6);
if ~isempty(opt.yLim), set(ax, 'YLim', opt.yLim); end

% Legend ordered by ranking
leg = legend(ax, 'Location','southoutside', 'NumColumns', min(3, max(1, ceil(N/4))));
leg.Box = 'off';

% ---- Save FIG ----
outDir = char(opt.outDir);           % normalize to char for exist/mkdir
if isempty(outDir), outDir = pwd; end

[ok,msg] = mkdir(outDir);            % creates parents if needed
if ~ok
    error('Could not create output directory "%s": %s', outDir, msg);
end

baseName = sprintf('performance_over_horizon_%s_all_datasets', lower(metric));
outFIG   = fullfile(outDir, [baseName '.fig']);

try
    savefig(fh, outFIG);
catch ME
    error('Failed to save FIG to "%s": %s', outFIG, ME.message);
end

fh.UserData.outputPathFIG = outFIG;
fh.UserData.summary = G;
fh.UserData.ranking = S;

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
