function fh = plot_model_comparison_box_by_country(csvFile, varargin)
% plot_model_comparison_box_by_country
% Boxplot comparison of MODELS for ONE country.
% Each box = distribution of the chosen metric across ALL horizons/rows
% for that country+model. White dot = mean.
%
% Example:
%   fh = plot_model_comparison_box_by_country('results_log_main.csv', ...
%           'metric','F2','country','France','outDir','figures');

% --------- Args ---------
p = inputParser;
addParameter(p, 'metric', 'F2', @(s)ischar(s) || isstring(s));
addParameter(p, 'country', '', @(s)ischar(s) || isstring(s));
addParameter(p, 'outDir', 'figures', @(s)ischar(s) || isstring(s));
addParameter(p, 'titleStr', '', @(s)ischar(s) || isstring(s));
addParameter(p, 'baseFontSize', 11, @isscalar);
addParameter(p, 'savePNG', true, @islogical);
addParameter(p, 'saveFIG', true, @islogical);
parse(p, varargin{:});
opt = p.Results;

metric   = char(opt.metric);
onlyCtry = strtrim(string(opt.country));
outDir   = char(opt.outDir);
baseFS   = opt.baseFontSize;

if strcmp(onlyCtry,"")
    error('Specify a country, e.g., ''France'', ''Italy'', or ''Greece''.');
end

% --------- Load & prep ---------
T = readtable(csvFile, 'TextType','string');

need = ["data_id","model","month",metric];
missing = setdiff(need, string(T.Properties.VariableNames));
if ~isempty(missing)
    error('Missing columns: %s', strjoin(missing, ', '));
end

% Country from data_id
T.country = extractBefore(T.data_id, "_");
T.country(T.country=="") = T.data_id(T.country==""); % fallback

% Make metric numeric
if ~isnumeric(T.(metric)), T.(metric) = str2double(string(T.(metric))); end
T = T(~isnan(T.(metric)), :);

% Filter to country
T = T(T.country==onlyCtry, :);
if isempty(T)
    error('No rows for country "%s".', onlyCtry);
end

% --------- Order models by mean metric (descending) ---------
G = groupsummary(T, 'model', 'mean', metric);
G.Mean = G.("mean_"+metric);
G = sortrows(G, 'Mean', 'descend');
models = G.model;     % ordered model list
N = numel(models);

% Colors (Tableau-20)
cols = tableau20_colors(min(N,20));

% --------- Assemble data for boxchart ---------
X = []; Y = []; idx = []; means = nan(N,1);
for i = 1:N
    msk  = T.model == models(i);
    vals = T.(metric)(msk);
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

% Draw boxes first
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

% Now compute y-limits and paint subtle alternating background bands
ylim(ax, [0 1]);   % <<< force y-axis from 0 to 1
yl = ylim(ax);
for i = 1:N
    colBand = (mod(i,2)==1) * [1 1 1] + (mod(i,2)==0) * [0.97 1.00 0.97];
    p = patch(ax, [i-0.5 i+0.5 i+0.5 i-0.5], [yl(1) yl(1) yl(2) yl(2)], ...
              colBand, 'EdgeColor','none', 'FaceAlpha',0.20);
    uistack(p,'bottom');
end

% Mean markers (white with thin black edge)
for i = 1:N
    scatter(ax, i, means(i), 28, 'w', 'filled', ...
        'MarkerEdgeColor', 'k', 'LineWidth', 0.7);
end

% Cosmetics
xlim(ax, [0.5 N+0.5]);
set(ax, 'XTick', 1:N, 'XTickLabel', models, ...
    'XTickLabelRotation', 25, 'FontSize', baseFS);
ylabel(ax, sprintf('%s (distribution across horizons)', metric), 'FontSize', baseFS);
ttl = sprintf('Model comparison — %s (%s)', metric, onlyCtry);
if ~isempty(opt.titleStr), ttl = opt.titleStr; end
title(ax, ttl, 'FontWeight','bold', 'FontSize', baseFS+2);

% --------- Save ---------
if ~exist(outDir, 'dir'); mkdir(outDir); end
baseName = sprintf('overall_box_%s_%s', lower(metric), onlyCtry);

% Finish any pending drawing before saving
drawnow;

% Make sure we have a valid figure handle
if ~ishandle(fh) || ~strcmp(get(fh, 'Type'), 'figure')
    fh = gcf;  % fallback to current figure, just in case
end

% Save FIG (editable)
figPath = fullfile(outDir, baseName + ".fig");

% Convert to char for maximum compatibility
figPath = char(figPath);

try
    savefig(fh, figPath);
catch ME
    warning('savefig failed (%s). Retrying with gcf...', ME.message);
    savefig(gcf, figPath);
end


% Expose summary
fh.UserData.country    = onlyCtry;
fh.UserData.metric     = metric;
fh.UserData.modelOrder = models;
fh.UserData.modelMeans = G.Mean;

end  % <<< end of main function

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
end  % <<< end of local function
