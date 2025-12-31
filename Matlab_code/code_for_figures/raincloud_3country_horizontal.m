function fh = raincloud_3country_horizontal(csvFile, varargin)
% One figure: 3 countries side-by-side (horizontal) + ONE shared legend
%
% Example:
% fh = raincloud_3country_horizontal('Final_results.csv', ...
%     'metric','F2', 'outDir','results', 'topN',10, ...
%     'highlightModels',["LWE","XGBoost","LightGBM"], ...
%     'baseFontSize',12, 'savePNG',true, 'saveFIG',true);

% -------- Args --------
p = inputParser;
addParameter(p,'metric','F2',@(s)ischar(s)||isstring(s));
addParameter(p,'outDir','results',@(s)ischar(s)||isstring(s));
addParameter(p,'topN',10,@isscalar);
addParameter(p,'highlightModels',["LWE"],@(x)isstring(x)||iscellstr(x));
addParameter(p,'baseFontSize',12,@isscalar);
addParameter(p,'savePNG',true,@islogical);
addParameter(p,'saveFIG',true,@islogical);
parse(p,varargin{:});
opt = p.Results;

metric = char(opt.metric);
outDir = char(opt.outDir);
hi = string(opt.highlightModels);

countries = ["France","Italy","Greece"];

% -------- Figure + horizontal layout --------
fh = figure('Color','w','Units','normalized', ...
    'Position',[0.05 0.20 0.90 0.55]);

tlo = tiledlayout(fh, 1, 3, ...
    'TileSpacing','compact', 'Padding','compact');

axesList = gobjects(3,1);

% -------- Draw each country panel --------
for i = 1:3
    ax = nexttile(tlo, i);
    axesList(i) = ax;

    % Draw temporary raincloud
    tmp = plot_raincloud_models_all_datasets(csvFile, ...
        'metric', metric, ...
        'country', countries(i), ...
        'topN', opt.topN, ...
        'highlightModels', hi, ...
        'outDir', outDir, ...
        'titleStr', '', ...
        'baseFontSize', opt.baseFontSize, ...
        'savePNG', false, ...
        'saveFIG', false);

    tmpAx = findobj(tmp, 'Type','axes');

    % Copy graphics
    copyobj(flipud(tmpAx.Children), ax);

    % Sync axes
    ax.XLim = tmpAx.XLim;
    ax.YLim = tmpAx.YLim;
    ax.YTick = tmpAx.YTick;
    ax.YTickLabel = tmpAx.YTickLabel;
    ax.FontSize = opt.baseFontSize;
    ax.Box = 'off';
    grid(ax,'on');

    % Panel label (top-left)
    text(ax, 0.02, 0.96, countries(i), ...
        'Units','normalized', ...
        'FontWeight','bold', ...
        'FontSize', opt.baseFontSize+1, ...
        'VerticalAlignment','top');

    % Only left panel keeps y-labels (journal standard)
    if i > 1
        ax.YTickLabel = [];
    end

    xlabel(ax, metric, 'FontSize', opt.baseFontSize);

    close(tmp);
end

% -------- One shared legend (dummy handles) --------
hold(axesList(1),'on');
hHi = plot(axesList(1), nan, nan, '-', 'LineWidth', 2.4);
hLo = plot(axesList(1), nan, nan, '-', 'LineWidth', 1.2);

hHi.Color = [0.10 0.45 0.80];   % highlight (LWE, top models)
hLo.Color = 0.65*[1 1 1];       % others (grey)

lg = legend(axesList(1), [hHi hLo], ...
    {'Highlighted models (LWE & top boosted trees)', 'Other models'}, ...
    'Location','southoutside', ...
    'NumColumns',2);

lg.Box = 'off';
lg.FontSize = opt.baseFontSize;

% -------- Global title --------
title(tlo, sprintf('Distribution of %s scores across models and countries', metric), ...
    'FontWeight','bold', ...
    'FontSize', opt.baseFontSize+2);

% -------- Save --------
if ~exist(outDir,'dir'); mkdir(outDir); end
baseName = "raincloud_" + lower(string(metric)) + "_3countries_horizontal";

if opt.saveFIG
    savefig(fh, fullfile(outDir, baseName + ".fig"));
end
if opt.savePNG
    exportgraphics(fh, fullfile(outDir, baseName + ".png"), 'Resolution', 300);
end

fh.UserData.output = fullfile(outDir, baseName + ".png");
end
