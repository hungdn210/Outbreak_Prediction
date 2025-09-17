function fh = plot_precision_recall_over_horizon(csvFile, varargin)
% plot_precision_recall_over_horizon
% Two-panel figure:
%   (Left)  Horizon (1..K) vs Recall   for each model
%   (Right) Horizon (1..K) vs Precision for each model
% If 'country' is empty, pools ALL datasets; otherwise filters to that country.
% Saves only a .fig for editing later.
%
% Example:
%   fh = plot_precision_recall_over_horizon('results_log_main.csv', ...
%           'country','France','outDir','figures','baseFontSize',11);

% ---------- Args ----------
p = inputParser;
addParameter(p, 'country', '', @(s)ischar(s) || isstring(s));     % '' => all datasets
addParameter(p, 'outDir', 'figures', @(s)ischar(s) || isstring(s));
addParameter(p, 'baseFontSize', 11, @isscalar);
addParameter(p, 'saveFIG', true, @islogical);
parse(p, varargin{:});
opt = p.Results;

onlyCtry = strtrim(string(opt.country));
outDir   = char(opt.outDir);
baseFS   = opt.baseFontSize;

% ---------- Load & guards ----------
T = readtable(csvFile, 'TextType','string');
need = ["data_id","model","month","Precision","Recall"];
missing = setdiff(need, string(T.Properties.VariableNames));
if ~isempty(missing)
    error('Missing required columns in CSV: %s', strjoin(missing, ', '));
end

% Country from data_id
T.country = extractBefore(T.data_id, "_");
T.country(T.country=="") = T.data_id(T.country==""); % fallback

% Numeric metrics
if ~isnumeric(T.Precision), T.Precision = str2double(string(T.Precision)); end
if ~isnumeric(T.Recall),    T.Recall    = str2double(string(T.Recall));    end
T = T(~isnan(T.Precision) & ~isnan(T.Recall), :);

% Horizon from 'Month+1'..'Month+6'
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
    if isempty(T), error('No rows for country "%s".', onlyCtry); end
end

% ---------- Aggregate: mean per (model,horizon) ----------
G = groupsummary(T, {'model','horizon'}, 'mean', {'Precision','Recall'});
G.P = G.mean_Precision;
G.R = G.mean_Recall;

% Horizons present
horizons = unique(G.horizon); horizons = sort(horizons);
H = numel(horizons);

% Models present (stable order)
models = unique(G.model, 'stable');
Nmods  = numel(models);

% Colors & markers
cols = hsv_distinct(Nmods);
markers = {'o','s','^','v','d','>','<','p','h','x','+'};

% Pre-build per-model series for both panels
Rmat = nan(H, Nmods);  % Recall
Pmat = nan(H, Nmods);  % Precision
for m = 1:Nmods
    sub = G(G.model==models(m), :);
    for k = 1:H
        hit = sub(sub.horizon==horizons(k), :);
        if ~isempty(hit)
            Rmat(k,m) = hit.R;
            Pmat(k,m) = hit.P;
        end
    end
end

% ---------- Figure ----------
fh = figure('Color','w','Units','normalized','Position',[0.08 0.15 0.80 0.58],'Visible','off');
tlo = tiledlayout(fh,1,2,'TileSpacing','compact','Padding','compact');

ttl = "Precision–Recall vs Forecast Horizon";
if strcmp(onlyCtry,"")
    ttl = ttl + " (ALL datasets)";
else
    ttl = ttl + " (" + onlyCtry + ")";
end
sgtitle(tlo, ttl, 'FontWeight','bold', 'FontSize', baseFS+2);

% ----- Left: Recall vs Horizon -----
ax1 = nexttile(tlo); hold(ax1,'on'); grid(ax1,'on'); box(ax1,'on');
for m = 1:Nmods
    mk = markers{mod(m-1,numel(markers))+1};
    plot(ax1, horizons, Rmat(:,m), '-o', ...
        'Color', cols(m,:), 'Marker', mk, ...
        'MarkerFaceColor', cols(m,:), 'MarkerEdgeColor',[0.2 0.2 0.2], ...
        'LineWidth', 1.8, 'MarkerSize', 5, ...
        'DisplayName', char(models(m)));
end
set(ax1,'XLim',[min(horizons)-0.2 max(horizons)+0.2], 'XTick', horizons, 'FontSize', baseFS);
ylim(ax1,[0 1]);
xlabel(ax1,'Forecast Horizon (Month +k)','FontSize',baseFS);
ylabel(ax1,'Recall','FontSize',baseFS);
title(ax1,'Recall vs Horizon','FontWeight','bold','FontSize',baseFS+1);

% ----- Right: Precision vs Horizon -----
ax2 = nexttile(tlo); hold(ax2,'on'); grid(ax2,'on'); box(ax2,'on');
for m = 1:Nmods
    mk = markers{mod(m-1,numel(markers))+1};
    plot(ax2, horizons, Pmat(:,m), '-o', ...
        'Color', cols(m,:), 'Marker', mk, ...
        'MarkerFaceColor', cols(m,:), 'MarkerEdgeColor',[0.2 0.2 0.2], ...
        'LineWidth', 1.8, 'MarkerSize', 5, ...
        'DisplayName', char(models(m)));
end
set(ax2,'XLim',[min(horizons)-0.2 max(horizons)+0.2], 'XTick', horizons, 'FontSize', baseFS);
ylim(ax2,[0 1]);
xlabel(ax2,'Forecast Horizon (Month +k)','FontSize',baseFS);
ylabel(ax2,'Precision','FontSize',baseFS);
title(ax2,'Precision vs Horizon','FontWeight','bold','FontSize',baseFS+1);

% One shared legend below panels
lgd = legend(ax2, 'Location','southoutside','NumColumns', max(3,ceil(Nmods/3)));
lgd.Box = 'off';

% ---------- Save FIG only ----------
if ~exist(outDir, 'dir'); mkdir(outDir); end
baseName = "prec_recall_over_horizon";
if ~strcmp(onlyCtry,"")
    baseName = baseName + "_" + onlyCtry;
else
    baseName = baseName + "_all_datasets";
end
figPath = char(fullfile(outDir, baseName + ".fig"));
if p.Results.saveFIG
    savefig(fh, figPath);
end

% Expose for downstream use
fh.UserData.country   = onlyCtry;
fh.UserData.horizons  = horizons;
fh.UserData.models    = models;
fh.UserData.Pmat      = Pmat;
fh.UserData.Rmat      = Rmat;
fh.UserData.figPath   = figPath;
end

% ---------- Local distinct HSV palette ----------
function C = hsv_distinct(n)
% n distinct colors around the hue circle, fixed saturation/value
if n <= 0, C = zeros(0,3); return; end
h = linspace(0, 1, n+1); h(end) = []; % drop duplicate at 1
s = 0.65; v = 0.9;
C = hsv2rgb([h(:), repmat(s,n,1), repmat(v,n,1)]);
end
