function fh = plot_model_comparison_box_all_datasets(csvFile, varargin)
% plot_model_comparison_box_all_datasets
% Boxplot comparison of MODELS pooled across ALL datasets (countries).
% Each box shows the distribution of the chosen metric across ALL rows
% (all countries × all horizons) for that model. White dot = mean.
%
% Example:
%   fh = plot_model_comparison_box_all_datasets('results_log_main.csv', ...
%       'metric','F2','outDir','figures');

% --------- Args ---------
p = inputParser;
addParameter(p, 'metric', 'F2', @(s)ischar(s) || isstring(s));
addParameter(p, 'outDir', 'figures', @(s)ischar(s) || isstring(s));
addParameter(p, 'titleStr', '', @(s)ischar(s) || isstring(s));
addParameter(p, 'baseFontSize', 11, @isscalar);
addParameter(p, 'savePNG', true, @islogical);
addParameter(p, 'saveFIG', true, @islogical);
parse(p, varargin{:});
opt = p.Results;

metric   = char(opt.metric);
outDir   = char(opt.outDir);
baseFS   = opt.baseFontSize;

% --------- Load & prep ---------
T = readtable(csvFile, 'TextType','string');
T = T( contains(T.data_id, "Greece"), : );

need = ["data_id","model","month",metric];
missing = setdiff(need, string(T.Properties.VariableNames));
if ~isempty(missing)
    error('Missing columns: %s', strjoin(missing, ', '));
end

% Country (not used for filtering here, but retained for reference)
T.country = extractBefore(T.data_id, "_");
T.country(T.country=="") = T.data_id(T.country==""); % fallback

% Metric numeric
if ~isnumeric(T.(metric)), T.(metric) = str2double(string(T.(metric))); end
T = T(~isnan(T.(metric)), :);

% --------- Order models by mean metric (descending) across ALL rows ----
G = groupsummary(T, 'model', 'mean', metric);
G.Mean = G.("mean_"+metric);
G = sortrows(G, 'Mean', 'descend');
models = G.model;
N = numel(models);
if N==0
    error('No models found in the table after cleaning.');
end

% Colors (Tableau-20)
cols = tableau20_colors(min(N,20));

% --------- Assemble data for boxchart ---------
X = []; Y = []; idx = []; means = nan(N,1);
for i = 1:N
    msk  = T.model == models(i);
    vals = T.(metric)(msk);            % all countries × all horizons for this model
    vals = vals(~isnan(vals));
    if isempty(vals), vals = NaN; end
    X   = [X; repmat(i, numel(vals), 1)];
    Y   = [Y; vals(:)];
    idx = [idx; repmat(i, numel(vals), 1)];
    means(i) = mean(vals, 'omitnan');
end

% --------- Plot ---------
fh = figure('Color','w','Units','normalized','Position',[0.06 0.15 0.88 0.65],'Visible','on');
ax = axes(fh); hold(ax,'on'); box(ax,'on'); grid(ax,'on');

% Draw boxes
for i = 1:N
    xi = X(idx==i); yi = Y(idx==i);
    boxchart(ax, xi, yi, ...
        'BoxFaceColor', cols(i,:), ...
        'BoxFaceAlpha', 0.85, ...
        'WhiskerLineColor', 0.25*[1 1 1], ...
        'MarkerStyle', '.', ...
        'MarkerColor', [0.15 0.15 0.15], ...
        'JitterOutliers', 'on');
end

% Fix y-axis to 0–1 and paint alternating background bands
ylim(ax, [0 1]);
% Mean markers (white with thin black edge)
for i = 1:N
    scatter(ax, i, means(i), 28, 'w', 'filled', ...
        'MarkerEdgeColor', 'k', 'LineWidth', 0.7);
end

% Cosmetics
xlim(ax, [0.5 N+0.5]);
set(ax, 'XTick', 1:N, 'XTickLabel', models, ...
    'XTickLabelRotation', 25, 'FontSize', baseFS);
ylabel(ax, "Average F2 (all datasets & horizons)", 'FontSize', baseFS);
ttl = opt.titleStr;
if isempty(ttl), ttl = sprintf('Model comparison — %s (ALL datasets)', metric); end
title(ax, ttl, 'FontWeight','bold', 'FontSize', baseFS+2);

% --------- Save ---------
if ~exist(outDir, 'dir'); mkdir(outDir); end
baseName = sprintf('overall_box_%s_all_datasets', lower(metric));
drawnow;

% Ensure valid figure handle
if ~ishandle(fh) || ~strcmp(get(fh,'Type'),'figure'), fh = gcf; end
figPath = char(fullfile(outDir, baseName + ".fig"));

if opt.saveFIG
    try, savefig(fh, figPath); catch, savefig(gcf, figPath); end
end

% Expose summary
fh.UserData.metric     = metric;
fh.UserData.modelOrder = models;
fh.UserData.modelMeans = G.Mean;

end

% ---------- local palette ----------
function C = tableau20_colors(n)
base = [ ...
     31 119 180; 255 127  14;  44 160  44; 214  39  40; 148 103 189; ...
    140  86  75; 227 119 194; 127 127 127; 188 189  34;  23 190 207; ...
    174 199 232; 255 187 120; 152 223 138; 255 152 150; 197 176 213; ...
    196 156 148; 247 182 210; 199 199 199; 219 219 141; 158 218 229 ...
] / 255;
n = min(n, size(base,1));
C = base(1:n,:);
end
