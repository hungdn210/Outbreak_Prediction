function fh = plot_rank_slopegraph_across_horizons(csvFile, varargin)
% plot_rank_slopegraph_across_horizons
% Slopegraph of MODEL RANKS across horizons (Month+1..Month+6).
% Rank is computed per horizon from the mean metric across ALL datasets.
% Lower rank = better (1 is best). Y-axis inverted (1 at top).
%
% Compatible with older MATLAB versions:
% - does NOT use MarkerFaceAlpha / MarkerEdgeAlpha (Line)
% - uses SCATTER for mean-rank points (more compatible)
% - does NOT use RGBA line colors (Color(4))

% ---- Parse args ----
p = inputParser;
addParameter(p, 'metric', 'F2', @(s)ischar(s) || isstring(s));
addParameter(p, 'outDir', 'figures', @(s)ischar(s) || isstring(s));
addParameter(p, 'titleStr', '', @(s)ischar(s) || isstring(s));
addParameter(p, 'baseFontSize', 11, @isscalar);
addParameter(p, 'lineWidth', 1.2, @isscalar);

addParameter(p, 'country', '', @(s)ischar(s) || isstring(s)); % '' => pooled, else France/Italy/Greece
addParameter(p, 'topN', inf, @isscalar);
addParameter(p, 'highlightModels', strings(0), @(x)isstring(x)||iscellstr(x));
addParameter(p, 'showLabels', true, @(b)islogical(b) && isscalar(b));
addParameter(p, 'savePNG', true, @(b)islogical(b) && isscalar(b));
addParameter(p, 'saveFIG', true, @(b)islogical(b) && isscalar(b));

parse(p, varargin{:});
opt = p.Results;

metric = char(opt.metric);
hiList = string(opt.highlightModels);
countryFilter = string(opt.country);

% ---- Read CSV ----
T = readtable(csvFile, 'TextType','string');

need = ["data_id","model","month",metric];
missing = setdiff(need, string(T.Properties.VariableNames));
if ~isempty(missing)
    error('Missing required columns in CSV: %s', strjoin(missing, ', '));
end

% ---- Country helper ----
T.country = extractBefore(T.data_id, "_");
T.country(T.country=="") = T.data_id(T.country=="");

if strlength(countryFilter) > 0
    T = T(T.country == countryFilter, :);
    if isempty(T)
        error('No rows match country="%s". Check data_id/country naming.', countryFilter);
    end
end

% ---- Horizon parse 'Month+1'..'Month+6' ----
mh = regexp(string(T.month),'Month\+(\d+)','tokens','once');
T.horizon = nan(height(T),1);
for i = 1:height(T)
    tok = mh{i};
    if ~isempty(tok)
        T.horizon(i) = str2double(tok{1});
    end
end
T = T(~isnan(T.horizon), :);

% metric numeric
if ~isnumeric(T.(metric))
    T.(metric) = str2double(string(T.(metric)));
end
T = T(~isnan(T.(metric)), :);

% ---- Aggregate mean metric per (model, horizon) across datasets ----
G = groupsummary(T, {'model','horizon'}, 'mean', metric);
G.Properties.VariableNames{end} = 'MeanMetric';

% ---- Overall mean to keep topN ----
S = groupsummary(G, 'model', 'mean', 'MeanMetric');
S.Properties.VariableNames{end} = 'OverallMean';
S = sortrows(S, 'OverallMean', 'descend');

if isfinite(opt.topN)
    keep = S.model(1:min(opt.topN, height(S)));
    G = G(ismember(G.model, keep), :);
    S = S(ismember(S.model, keep), :);
end

models = S.model;
N = numel(models);
K = max(G.horizon);

% ---- Compute rank per horizon ----
rankMat = nan(N, K);
for k = 1:K
    sub = G(G.horizon==k, :);

    [tf, loc] = ismember(models, sub.model);
    vals = nan(N,1);
    vals(tf) = sub.MeanMetric(loc(tf));

    tmp = vals;
    tmp(isnan(tmp)) = -inf;
    [~, ord] = sort(tmp, 'descend');

    r = nan(N,1);
    r(ord) = 1:N;
    rankMat(:,k) = r;
end

meanRank = mean(rankMat, 2, 'omitnan');

% ---- Figure ----
fh = figure('Color','w','Units','normalized','Position',[0.12 0.15 0.75 0.60], 'Visible','on');
ax = axes(fh); hold(ax,'on'); box(ax,'on'); grid(ax,'on');

xH = 1:K;
xMean = K + 1;

% styles
greyLine = [0.70 0.70 0.70];
greyDot  = [0.55 0.55 0.55];

hiColors = [ ...
    31 119 180;
    255 127  14;
    44 160  44;
    214  39  40;
    148 103 189;
    140  86  75;
    227 119 194;
    127 127 127;
    188 189  34;
    23 190 207] / 255;

hiColorMap = containers.Map('KeyType','char','ValueType','any');
for i = 1:numel(hiList)
    hiColorMap(char(hiList(i))) = hiColors( mod(i-1,size(hiColors,1)) + 1, : );
end

% ---- Plot all models ----
for i = 1:N
    mname = string(models(i));
    y = rankMat(i, :);
    yMean = meanRank(i);

    isHi = ismember(mname, hiList);

    if isHi
        col = hiColorMap(char(mname));
        lw = max(2.2, 1.8*opt.lineWidth);
    else
        col = greyLine;
        lw = max(0.7, 0.9*opt.lineWidth);
    end

    % line across horizons (no alpha for compatibility)
    plot(ax, xH, y, '-', 'Color', col, 'LineWidth', lw);

    % mean-rank point on right: use SCATTER (more compatible for alpha)
    if ~isnan(yMean)
        if isHi
            scatter(ax, xMean, yMean, 38, col, 'filled');
        else
            % if your MATLAB supports MarkerFaceAlpha for scatter, use it; else it will just ignore
            s = scatter(ax, xMean, yMean, 28, greyDot, 'filled');
            try
                s.MarkerFaceAlpha = 0.35;
            catch
                % older versions: ignore alpha
            end
        end
    end

    % right-side labels for highlighted models
    if opt.showLabels && isHi && ~isnan(yMean)
        text(ax, xMean + 0.12, yMean, char(mname), ...
            'FontSize', opt.baseFontSize, ...
            'Color', col, ...
            'FontWeight','bold', ...
            'VerticalAlignment','middle');
    end
end

% ---- Axes formatting ----
xlabel(ax, 'Forecast horizon (Month +k)   \rightarrow   Mean rank', 'FontSize', opt.baseFontSize);
ylabel(ax, 'Rank (1 = best)', 'FontSize', opt.baseFontSize);

ttl = string(opt.titleStr);
if strlength(ttl) == 0
    if strlength(countryFilter) > 0
        ttl = countryFilter + " — Rank stability across horizons (" + metric + ")";
    else
        ttl = "All datasets — Rank stability across horizons (" + metric + ")";
    end
end
title(ax, char(ttl), 'FontWeight','bold', 'FontSize', opt.baseFontSize+2);

set(ax, 'YDir','reverse');

xt = [1:K, xMean];
xtlbl = [compose('Month+%d', 1:K), "Mean rank"];
set(ax, 'XLim',[0.8 xMean+1.2], 'XTick', xt, 'XTickLabel', xtlbl);

maxTicks = min(N, 15);
yt = unique(round(linspace(1, N, maxTicks)));
set(ax, 'YLim',[0.5 N+0.5], 'YTick', yt);

set(ax, 'FontSize', opt.baseFontSize);

xline(ax, K+0.5, ':', 'Color', [0.35 0.35 0.35], 'LineWidth', 1.0);

% ---- Save outputs ----
outDir = char(opt.outDir);
if isempty(outDir), outDir = pwd; end
if ~exist(outDir, 'dir'); mkdir(outDir); end

tag = "all";
if strlength(countryFilter) > 0, tag = lower(countryFilter); end

baseName = sprintf('rank_slopegraph_%s_%s', lower(metric), tag);
outFIG = fullfile(outDir, [baseName '.fig']);
outPNG = fullfile(outDir, [baseName '.png']);

if opt.saveFIG
    savefig(fh, outFIG);
end
if opt.savePNG
    try
        exportgraphics(fh, outPNG, 'Resolution', 350);
    catch
        % older MATLAB fallback
        print(fh, outPNG, '-dpng', '-r350');
    end
end

fh.UserData.ranks = rankMat;
fh.UserData.meanRank = meanRank;
fh.UserData.models = models;
fh.UserData.metric = metric;
fh.UserData.country = char(countryFilter);
fh.UserData.outputFIG = outFIG;
fh.UserData.outputPNG = outPNG;

end
