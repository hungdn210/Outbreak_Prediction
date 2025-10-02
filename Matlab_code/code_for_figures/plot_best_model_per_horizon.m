function fh = plot_best_model_per_horizon(csvFile, varargin)
% plot_best_model_per_horizon
% Bar chart of the best model (highest mean metric) at each forecast horizon.
% If 'country' is empty, pools across ALL datasets. Otherwise filters to that country.
%
% Example:
%   fh = plot_best_model_per_horizon('results_log_main.csv', ...
%           'metric','F2','country','', 'outDir','figures');

% --------- Args ---------
p = inputParser;
addParameter(p, 'metric', 'F2', @(s)ischar(s) || isstring(s));
addParameter(p, 'country', '', @(s)ischar(s) || isstring(s));     % '' => all datasets
addParameter(p, 'outDir', 'figures', @(s)ischar(s) || isstring(s));
addParameter(p, 'baseFontSize', 11, @isscalar);
addParameter(p, 'savePNG', true, @islogical);
addParameter(p, 'saveFIG', true, @islogical);
parse(p, varargin{:});
opt = p.Results;

metric   = char(opt.metric);
onlyCtry = strtrim(string(opt.country));
outDir   = char(opt.outDir);
baseFS   = opt.baseFontSize;

% --------- Load & prep ---------
T = readtable(csvFile, 'TextType','string');

need = ["data_id","model","month",metric];
missing = setdiff(need, string(T.Properties.VariableNames));
if ~isempty(missing)
    error('Missing columns in CSV: %s', strjoin(missing, ', '));
end

% Country from data_id
T.country = extractBefore(T.data_id, "_");
T.country(T.country=="") = T.data_id(T.country==""); % fallback

% Metric numeric
if ~isnumeric(T.(metric)), T.(metric) = str2double(string(T.(metric))); end
T = T(~isnan(T.(metric)), :);

% Horizon number from 'Month+1' ... 'Month+6'
tok = regexp(string(T.month),'Month\+(\d+)','tokens','once');
h = nan(height(T),1);
for i = 1:height(T)
    if ~isempty(tok{i}), h(i) = str2double(tok{i}{1}); end
end
T.horizon = h;
T = T(~isnan(T.horizon), :);

% Optional country filter
if ~strcmp(onlyCtry,"")
    T = T(T.country==onlyCtry, :);
    if isempty(T), error('No rows after filtering for country "%s".', onlyCtry); end
end

% --------- Average per (model,horizon) and pick winners ---------
G = groupsummary(T, {'model','horizon'}, 'mean', metric);
G.MeanMetric = G.("mean_"+metric);

horizons = unique(G.horizon);
horizons = sort(horizons);
H = numel(horizons);

winners = strings(H,1);
winVals = nan(H,1);

for k = 1:H
    sub = G(G.horizon==horizons(k), :);
    [winVals(k), idx] = max(sub.MeanMetric);
    winners(k) = sub.model(idx);
end

% --------- Color map for winners (consistent & distinct) ---------
winModels = unique(winners, 'stable');               % models that actually win at least once
colsAll   = tableau20_colors(20);
% map each winner model to a color index (stable)
keys  = cellstr(winModels);
vals  = num2cell(1:numel(winModels));
m2c   = containers.Map(keys, vals);

barColors = zeros(H,3);
for k = 1:H
    barColors(k,:) = colsAll(m2c(char(winners(k))), :);
end

% --------- Plot ---------
fh = figure('Color','w','Units','normalized','Position',[0.15 0.2 0.6 0.5],'Visible','on');
ax = axes(fh); hold(ax,'on'); box(ax,'on'); grid(ax,'on');

% Use categorical x with Month+1..K labels
xCats = categorical(string("Month+" + horizons));
xCats = reordercats(xCats, string("Month+" + horizons));

b = bar(ax, xCats, winVals, 0.65, 'FaceColor','flat', 'EdgeColor',[0.25 0.25 0.25]);
b.CData = barColors;

% annotations: value text on top of bars
for k = 1:H
    text(ax, k, winVals(k)+0.02, sprintf('%.3f', winVals(k)), ...
        'HorizontalAlignment','center', 'FontSize', baseFS-1);
end

% y-range & labels
ylim(ax, [0 1]);
ylabel(ax, sprintf('Best %s (mean per horizon)', metric), 'FontSize', baseFS);

% title
if strcmp(onlyCtry,"")
    ttl = sprintf('Best model per horizon — %s (ALL datasets)', metric);
else
    ttl = sprintf('Best model per horizon — %s (%s)', metric, onlyCtry);
end
title(ax, ttl, 'FontWeight','bold', 'FontSize', baseFS+2);

set(ax, 'FontSize', baseFS);

% legend of winner models (color matches bars)
hold(ax,'on');
lgd = legend(ax);
lgd.String = cellstr(winModels);
lgd.ItemTokenSize = [12 12];
lgd.Location = 'southoutside';
lgd.NumColumns = max(2, ceil(numel(winModels)/3));
lgd.Box = 'off';

% --------- Save ---------
if ~exist(outDir, 'dir'); mkdir(outDir); end
baseName = sprintf('best_model_per_horizon_%s', lower(metric));
if ~strcmp(onlyCtry,"")
    baseName = baseName + "_" + onlyCtry;
else
    baseName = baseName + "_all_datasets";
end
figPath = char(fullfile(outDir, baseName + ".fig"));

drawnow;
if p.Results.saveFIG
    try, savefig(fh, figPath); catch, savefig(gcf, figPath); end
end

% expose results for downstream use
fh.UserData.horizons = horizons;
fh.UserData.winners  = winners;
fh.UserData.values   = winVals;

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
