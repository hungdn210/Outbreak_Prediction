function fh = plot_top3_models_per_horizon(csvFile, varargin)
% plot_top3_models_per_horizon
% Grouped bar chart of the TOP 3 models per horizon (by mean metric).
% Saves only a .fig file (editable in MATLAB).
%
% Example:
%   fh = plot_top3_models_per_horizon('results_log_main.csv', ...
%            'metric','F2','country','France','outDir','figures');

% --------- Args ---------
p = inputParser;
addParameter(p, 'metric', 'F2', @(s)ischar(s) || isstring(s));
addParameter(p, 'country', '', @(s)ischar(s) || isstring(s));   % '' = all datasets
addParameter(p, 'outDir', 'figures', @(s)ischar(s) || isstring(s));
addParameter(p, 'baseFontSize', 11, @isscalar);
parse(p, varargin{:});
opt = p.Results;

metric   = char(opt.metric);
onlyCtry = strtrim(string(opt.country));
outDir   = char(opt.outDir);
baseFS   = opt.baseFontSize;

% --------- Load data ---------
T = readtable(csvFile, 'TextType','string');

% Guard
need = ["data_id","model","month",metric];
missing = setdiff(need, string(T.Properties.VariableNames));
if ~isempty(missing)
    error('Missing required columns: %s', strjoin(missing, ', '));
end

% Country extraction
T.country = extractBefore(T.data_id, "_");
T.country(T.country=="") = T.data_id(T.country==""); % fallback

% Numeric metric
if ~isnumeric(T.(metric)), T.(metric) = str2double(string(T.(metric))); end
T = T(~isnan(T.(metric)), :);

% Horizon number
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
    if isempty(T)
        error('No rows for country "%s".', onlyCtry);
    end
end

% --------- Aggregate: mean per (model,horizon) ---------
G = groupsummary(T, {'model','horizon'}, 'mean', metric);
G.MeanMetric = G.("mean_"+metric);

horizons = unique(G.horizon);
horizons = sort(horizons);
H = numel(horizons);

% --------- Top 3 per horizon ---------
rows = {};
for k = 1:H
    sub = G(G.horizon==horizons(k), :);
    sub = sortrows(sub, 'MeanMetric', 'descend');
    topN = min(3, height(sub));
    for r = 1:topN
        rows(end+1,:) = {horizons(k), r, strtrim(sub.model(r)), sub.MeanMetric(r)}; %#ok<AGROW>
    end
end
R = cell2table(rows, 'VariableNames', {'Horizon','Rank','Model','Value'});

% --------- Plotting ---------
fh = figure('Color','w','Units','normalized','Position',[0.1 0.15 0.75 0.55],'Visible','off');
ax = axes(fh); hold(ax,'on'); box(ax,'on'); grid(ax,'on');

% Models that ever appear in top-3 (stable order)
uniqueModels = unique(R.Model, 'stable');

% Create a large distinct color map (HSV evenly spaced)
Nmods = numel(uniqueModels);
colsAll = hsv_distinct(Nmods);   % <- robust palette

% Assign stable color to each model
keys  = cellstr(uniqueModels);
vals  = num2cell(1:Nmods);
m2c   = containers.Map(keys, vals);

% Build data matrix (H x 3)
Y = nan(H,3);
modelNames = strings(H,3);
for k = 1:H
    sub = R(R.Horizon==horizons(k), :);
    for r = 1:height(sub)
        Y(k,sub.Rank(r)) = sub.Value(r);
        modelNames(k,sub.Rank(r)) = sub.Model(r);
    end
end

% Plot bars individually so we can color them by model.
% Also: only the FIRST bar for each model is visible to legend.
barW = 0.8;                            %#ok<NASGU> % (kept for clarity)
groupX = 1:H;
offsets = [-0.25, 0, 0.25];            % for 3 ranks per group
plottedOnce = containers.Map('KeyType','char','ValueType','logical');

for r = 1:3
    for k = 1:H
        if ~isnan(Y(k,r))
            m = strtrim(char(modelNames(k,r)));
            ci = m2c(m);
            col = colsAll(ci,:);

            % Only first bar of a model appears in legend
            if ~isKey(plottedOnce, m) || ~plottedOnce(m)
                hBar = bar(groupX(k)+offsets(r), Y(k,r), 0.25, ...
                    'FaceColor', col, 'EdgeColor',[0.25 0.25 0.25], ...
                    'DisplayName', m);                   % legend entry
                plottedOnce(m) = true;
            else
                bar(groupX(k)+offsets(r), Y(k,r), 0.25, ...
                    'FaceColor', col, 'EdgeColor',[0.25 0.25 0.25], ...
                    'HandleVisibility','off');           % no legend entry
            end

            % annotate
            text(groupX(k)+offsets(r), Y(k,r)+0.02, sprintf('%.3f', Y(k,r)), ...
                'HorizontalAlignment','center', 'FontSize', baseFS-1);
        end
    end
end

% Cosmetics
set(ax,'XTick',1:H, 'XTickLabel',"Month+" + horizons, 'FontSize',baseFS);
ylabel(ax, sprintf('Top-3 %s per horizon', metric), 'FontSize', baseFS);
ylim(ax,[0 1]);
ttl = sprintf('Top-3 models per horizon — %s', metric);
if ~strcmp(onlyCtry,""), ttl = ttl + " (" + onlyCtry + ")"; else, ttl = ttl + " (ALL datasets)"; end
title(ax, ttl, 'FontWeight','bold','FontSize', baseFS+2);

% Legend (one entry per model, unique colors)
lgd = legend(ax, 'Location','southoutside','NumColumns',max(3,ceil(Nmods/4)));
lgd.Box = 'off';

% --------- Save FIG only ---------
if ~exist(outDir, 'dir'); mkdir(outDir); end
baseName = sprintf('top3_models_per_horizon_%s', lower(metric));
if ~strcmp(onlyCtry,"")
    baseName = baseName + "_" + onlyCtry;
else
    baseName = baseName + "_all_datasets";
end
figPath = char(fullfile(outDir, baseName + ".fig"));
savefig(fh, figPath);

% Expose results
fh.UserData.table = R;
fh.UserData.horizons = horizons;
fh.UserData.uniqueModels = uniqueModels;

end

% ---------- local distinct HSV palette ----------
function C = hsv_distinct(n)
% n distinct colors around the hue circle, fixed saturation/value
if n <= 0
    C = zeros(0,3);
    return;
end
h = linspace(0, 1, n+1); h(end) = []; % drop duplicate at 1
s = 0.65; v = 0.9;
C = hsv2rgb([h(:), repmat(s,n,1), repmat(v,n,1)]);
end
