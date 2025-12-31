function fh = plot_raincloud_models_all_datasets(csvFile, varargin)
% plot_raincloud_models_all_datasets
% Raincloud plot per model: half-violin (density) + box + jittered points.
% Pooled across all datasets/horizons (or filter by country if provided).
%
% CHANGE: only highlightModels get DARK colors; all other models get LIGHT pastel colors.

% ---------------- Args ----------------
p = inputParser;
addParameter(p,'metric','F2',@(s)ischar(s)||isstring(s));
addParameter(p,'outDir','figures',@(s)ischar(s)||isstring(s));
addParameter(p,'titleStr','',@(s)ischar(s)||isstring(s));
addParameter(p,'baseFontSize',11,@isscalar);
addParameter(p,'country','',@(s)ischar(s)||isstring(s));         % '' => all
addParameter(p,'topN',inf,@isscalar);                            % show top-N by mean
addParameter(p,'highlightModels',strings(0),@(x)isstring(x)||iscellstr(x));
addParameter(p,'savePNG',true,@islogical);
addParameter(p,'saveFIG',true,@islogical);
addParameter(p,'bandwidth',[],@(v)isnumeric(v)&& (isempty(v)||isscalar(v)));
addParameter(p,'jitterWidth',0.18,@isscalar);
addParameter(p,'violinWidth',0.35,@isscalar);
parse(p,varargin{:});
opt = p.Results;

metric = char(opt.metric);
outDir = char(opt.outDir);
hiList = string(opt.highlightModels);

% ---------------- Read & prep ----------------
T = readtable(csvFile,'TextType','string');

need = ["data_id","model","month",metric];
missing = setdiff(need, string(T.Properties.VariableNames));
if ~isempty(missing)
    error('Missing required columns: %s', strjoin(missing, ', '));
end

% Country label
T.country = extractBefore(T.data_id,"_");
T.country(T.country=="") = T.data_id(T.country=="");

if strlength(string(opt.country))>0
    T = T(T.country == string(opt.country), :);
end

% Metric numeric
if ~isnumeric(T.(metric))
    T.(metric) = str2double(string(T.(metric)));
end
T = T(isfinite(T.(metric)), :);

% Rank models by mean
G = groupsummary(T,'model','mean',metric);
G.Mean = G.("mean_"+metric);
G = sortrows(G,'Mean','descend');

if isfinite(opt.topN)
    keep = G.model(1:min(height(G), opt.topN));
    T = T(ismember(T.model, keep), :);
    G = G(ismember(G.model, keep), :);
end

models = G.model;
N = numel(models);

% ---- palettes ----
lightCols = pastel15_colors();              % all non-highlight models
darkCols  = dark10_colors();                % used only for highlight models
hiList    = unique(hiList, 'stable');
hiList    = hiList(hiList~="");             % guard empty strings

% map highlight model -> dark color (stable assignment)
hiToColor = containers.Map('KeyType','char','ValueType','any');
for j = 1:numel(hiList)
    hiToColor(char(hiList(j))) = darkCols(mod(j-1,size(darkCols,1))+1,:);
end

% ---------------- Plot ----------------
fh = figure('Color','w','Units','normalized','Position',[0.08 0.10 0.86 0.75]);
ax = axes(fh); hold(ax,'on'); box(ax,'off'); grid(ax,'on');

% IMPORTANT: YTick must be increasing; use axis YDir reverse to put best on top.
ypos = 1:N;                 % increasing
ylabels = string(models);   % already ranked best -> worst

% global x-limits
xAll = T.(metric);
xMin = max(0, min(xAll) - 0.03);
xMax = min(1, max(xAll) + 0.03);

% deterministic jitter
rng(1);

% draw each raincloud row
for i = 1:N
    m = models(i);
    vals = T.(metric)(T.model==m);
    if isempty(vals), continue; end

    y0 = ypos(i);

    % ---- styling: DARK only for highlight models ----
    isHi = ismember(string(m), hiList);

    if isHi
        col = hiToColor(char(m));   % dark
        lwMed  = 2.2;
        vAlpha = 0.32;
        ptSize = 12;
    else
        col = lightCols(mod(i-1,size(lightCols,1))+1,:);  % light
        lwMed  = 1.1;
        vAlpha = 0.16;
        ptSize = 9;
    end

    % ---------------- KDE / half violin ----------------
    valsKDE = vals(:);
    valsKDE = valsKDE(isfinite(valsKDE));
    valsKDE = max(0, min(1, valsKDE));   % clamp to [0,1]

    if numel(valsKDE) >= 5
        sup = [-1e-6, 1+1e-6]; % tolerate float edge cases
        try
            if isempty(opt.bandwidth)
                [f,xi] = ksdensity(valsKDE, 'Support', sup, 'BoundaryCorrection','reflection');
            else
                [f,xi] = ksdensity(valsKDE, 'Support', sup, 'BoundaryCorrection','reflection', ...
                                   'Bandwidth', opt.bandwidth);
            end
        catch
            if isempty(opt.bandwidth)
                [f,xi] = ksdensity(valsKDE);
            else
                [f,xi] = ksdensity(valsKDE, 'Bandwidth', opt.bandwidth);
            end
        end

        xi = max(0, min(1, xi));
        f  = f ./ (max(f) + eps);

        width = opt.violinWidth;
        yy = y0 + f*width;
        xx = xi;

        patch(ax, [xx fliplr(xx)], [yy fliplr(y0*ones(size(yy)))], col, ...
            'FaceAlpha', vAlpha, 'EdgeColor', 'none');
    end

    % ---------------- jittered raw points ----------------
    jitter = (rand(size(vals)) - 0.5) * opt.jitterWidth;

    if isHi
        ptAlpha = 0.28;
    else
        ptAlpha = 0.10;
    end

    sc = scatter(ax, vals, y0 + jitter, ptSize, 'filled', ...
        'MarkerFaceColor', col, ...
        'MarkerEdgeColor', 'none');
    % alpha (works for scatter)
    try
        sc.MarkerFaceAlpha = ptAlpha;
    catch
        % older MATLAB: ignore alpha silently
    end

    % ---------------- box (median/IQR) ----------------
    q = quantile(vals, [0.25 0.5 0.75]);
    iqrL = q(3)-q(1);

    lo = max(min(vals), q(1)-1.5*iqrL);
    hi = min(max(vals), q(3)+1.5*iqrL);

    plot(ax, [lo hi], [y0 y0], '-', 'Color', 0.30*[1 1 1], 'LineWidth', 0.9);

    rectX = q(1);
    rectW = max(1e-6, q(3)-q(1));
    rectangle(ax, 'Position', [rectX, y0-0.12, rectW, 0.24], ...
        'FaceColor', [col 0.18], 'EdgeColor', col, 'LineWidth', 0.9);

    plot(ax, [q(2) q(2)], [y0-0.12 y0+0.12], '-', 'Color', 'k', 'LineWidth', lwMed);

    mu = mean(vals,'omitnan');
    plot(ax, mu, y0, 'o', 'MarkerSize', 5, 'MarkerFaceColor', 'w', ...
        'MarkerEdgeColor', 'k', 'LineWidth', 0.8);
end

% ---------------- axes cosmetics ----------------
set(ax,'YTick',ypos,'YTickLabel',ylabels,'FontSize',opt.baseFontSize);
set(ax,'YDir','reverse');  % best model (i=1) appears at top
xlabel(ax, sprintf('%s (pooled across horizons)', metric), 'FontSize', opt.baseFontSize);
xlim(ax, [xMin xMax]);
ylim(ax, [0.5 N+0.5]);

ttl = string(opt.titleStr);
if strlength(ttl)==0
    if strlength(string(opt.country))>0
        ttl = string(opt.country) + " — Raincloud comparison";
    else
        ttl = "All datasets — Raincloud comparison";
    end
end
title(ax, ttl, 'FontWeight','bold', 'FontSize', opt.baseFontSize+2);

% ---------------- legend (clean) ----------------
hDark = plot(ax, nan, nan, '-', 'Color', darkCols(1,:), 'LineWidth', 2.2);
hLite = plot(ax, nan, nan, '-', 'Color', lightCols(1,:), 'LineWidth', 1.2);
hMu   = plot(ax, nan, nan, 'o', 'MarkerFaceColor','w', 'MarkerEdgeColor','k', ...
            'MarkerSize', 6, 'LineWidth', 0.8);
hMd   = plot(ax, nan, nan, '-', 'Color', 'k', 'LineWidth', 2.0);

lg = legend(ax, [hDark hLite hMu hMd], ...
    {'Highlighted models', 'Other models', 'Mean', 'Median'}, ...
    'Location','southoutside', 'NumColumns',4);
lg.Box = 'off';
lg.FontSize = opt.baseFontSize;

% ---------------- save ----------------
if ~exist(outDir,'dir'); mkdir(outDir); end
baseName = "raincloud_" + lower(string(metric));
if strlength(string(opt.country))>0
    baseName = baseName + "_" + lower(string(opt.country));
else
    baseName = baseName + "_all";
end

fh.UserData.metric = metric;
fh.UserData.modelOrder = models;
fh.UserData.means = G.Mean;

if opt.saveFIG
    savefig(fh, fullfile(outDir, baseName + ".fig"));
end
if opt.savePNG
    exportgraphics(fh, fullfile(outDir, baseName + ".png"), 'Resolution', 300);
    fh.UserData.outputPNG = fullfile(outDir, baseName + ".png");
end
end

% ---------- palettes ----------
function C = pastel15_colors()
% Soft pastel palette (15) in [0,1]
C = [
    0.65, 0.89, 0.61;
    0.70, 0.85, 0.95;
    0.98, 0.80, 0.80;
    0.95, 0.90, 0.65;
    0.85, 0.75, 0.90;
    0.98, 0.85, 0.65;
    0.80, 0.90, 0.90;
    0.90, 0.85, 0.80;
    0.85, 0.85, 0.85;
    0.75, 0.88, 0.75;
    0.85, 0.80, 0.95;
    0.95, 0.75, 0.85;
    0.80, 0.85, 0.95;
    0.90, 0.95, 0.80;
    0.95, 0.90, 0.85;
];
end

function C = dark10_colors()
% Darker, still publication-friendly (10) in [0,1]
C = [
    0.10, 0.45, 0.80;  % blue
    0.85, 0.33, 0.10;  % orange
    0.15, 0.60, 0.20;  % green
    0.70, 0.20, 0.20;  % red
    0.45, 0.30, 0.65;  % purple
    0.35, 0.35, 0.35;  % dark gray
    0.55, 0.40, 0.25;  % brown
    0.85, 0.45, 0.70;  % pink
    0.40, 0.65, 0.75;  % teal
    0.55, 0.55, 0.20;  % olive
];
end
