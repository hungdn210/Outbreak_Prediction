function fh = plot_all_models_hist(csvFile, varargin)
% plot_all_models_hist
% Histogram-style bar chart: one bar per model.
% Saves a .fig file (like the top3 function).
%
% Example:
% fh = plot_all_models_hist('results.csv', ...
%           'metric','F2','outDir','figures');

% ----------- Args -----------
p = inputParser;
addParameter(p,'metric','F2',@(s)ischar(s)||isstring(s));
addParameter(p,'outDir','figures',@(s)ischar(s)||isstring(s));
addParameter(p,'baseFontSize',12,@isscalar);
parse(p,varargin{:});
opt = p.Results;

metric = char(opt.metric);
outDir = char(opt.outDir);
fs = opt.baseFontSize;

% ----------- Read CSV -----------
T = readtable(csvFile,'TextType','string');

% Require: model + metric
need = ["model", metric];
missing = setdiff(need, string(T.Properties.VariableNames));
if ~isempty(missing)
    error("Missing required columns: %s", strjoin(missing,', '));
end

% Convert metric to numeric
if ~isnumeric(T.(metric))
    T.(metric) = str2double(string(T.(metric)));
end
T = T(~isnan(T.(metric)),:);

% ----------- Aggregate mean metric per model -----------
G = groupsummary(T, "model", "mean", metric);
G.MeanMetric = G.("mean_"+metric);

% Sort best → worst
G = sortrows(G,"MeanMetric","descend");

% ----------- Make figure -----------
fh = figure('Color','w','Units','normalized', ...
            'Position',[0.2 0.15 0.6 0.55],'Visible','on');

ax = axes(fh); hold(ax,'on'); grid(ax,'on'); box(ax,'on');

% Colors
N = height(G);
colors = hsv_distinct(N);

% Horizontal bars
% Horizontal bars: one bar per model at y = i
hb = gobjects(N,1);

for i = 1:N
    hb(i) = barh(ax, i, G.MeanMetric(i), 0.6, ...  % <-- note the 'i' and width
        'FaceColor', colors(i,:), ...
        'EdgeColor','none', ...
        'DisplayName', G.model{i});   % legend label
end

% Tidy axes
yticks(1:N);
yticklabels(G.model);
ylim([0 N+1]);

legend(ax,'Location','southoutside','NumColumns',2,'Box','off');
% Labels
yticks(1:N);
yticklabels(G.model);
xlabel(['Mean ', metric], 'FontSize', fs);
title(['Overall Model Performance (Mean ', metric, ')'], ...
      'FontWeight','bold','FontSize', fs+2);

set(ax,'YDir','reverse','FontSize',fs);

% Annotate values
for i = 1:N
    text(G.MeanMetric(i) + 0.005, i, sprintf('%.3f', G.MeanMetric(i)), ...
         'VerticalAlignment','middle','FontSize',fs-1);
end

% ----------- Save FIG -----------
if ~exist(outDir,'dir')
    mkdir(outDir);
end

figName = sprintf('all_models_mean_%s.fig', lower(metric));
figPath = fullfile(outDir, figName);
savefig(fh, figPath);

% Return useful data
fh.UserData.table  = G;
fh.UserData.output = figPath;

end

% ---------- Helper: distinct colors ----------
function C = hsv_distinct(n)
h = linspace(0,1,n+1); h(end) = [];
s = 0.65; v = 0.9;
C = hsv2rgb([h(:), repmat(s,n,1), repmat(v,n,1)]);
end
