function fh = plot_horizon_heatmap_all_dataset(csvFile, varargin)
% Heatmap of model performance per horizon (rows=models, cols=horizons)
% Aggregates across ALL datasets. Best models appear at the TOP.

% ---------- Args ----------
p = inputParser;
addParameter(p,'metric','F2',@(s)ischar(s)||isstring(s));
addParameter(p,'mode','score',@(s)any(validatestring(s,{'score','rank'}))); % 'score'|'rank'
addParameter(p,'outDir','figures',@(s)ischar(s)||isstring(s));
addParameter(p,'titleStr','',@(s)ischar(s)||isstring(s));
addParameter(p,'baseFontSize',11,@isscalar);
addParameter(p,'precision',3,@isscalar);
addParameter(p,'clim',[],@(v)isnumeric(v) && (isempty(v)||numel(v)==2));
parse(p,varargin{:});
opt = p.Results; metric = char(opt.metric);

% ---------- Load & guard ----------
T = readtable(csvFile, 'TextType','string', 'VariableNamingRule','preserve');
need = ["data_id","model","month",metric];
missing = setdiff(need, string(T.Properties.VariableNames));
if ~isempty(missing), error('Missing required columns: %s', strjoin(missing, ', ')); end

% ---------- Parse horizon from 'Month+K' ----------
mh = regexp(string(T.month),'Month\+(\d+)','tokens','once');
T.horizon = nan(height(T),1);
for i = 1:height(T)
    if ~isempty(mh{i}), T.horizon(i) = str2double(mh{i}{1}); end
end
T = T(~isnan(T.horizon),:);

if ~isnumeric(T.(metric)), T.(metric) = str2double(string(T.(metric))); end
T = T(~isnan(T.(metric)),:);

% ---------- Aggregate mean across datasets at (model,horizon) ----------
G = groupsummary(T, {'model','horizon'}, 'mean', metric);
G.Properties.VariableNames{end} = 'MeanMetric';

% ---------- Pivot to matrix ----------
W = unstack(G, 'MeanMetric', 'horizon');      % one row per model
modelNames = string(W.model);
W.model = [];

% Robustly extract horizon numbers from column names
vnames = string(W.Properties.VariableNames);
hnum = nan(size(vnames));
for k = 1:numel(vnames)
    tok = regexp(vnames(k), '(\d+)$', 'tokens', 'once'); % trailing digits
    if ~isempty(tok), hnum(k) = str2double(tok{1}); end
end
valid = ~isnan(hnum);
if ~any(valid), error('Could not parse horizons from: %s', strjoin(vnames, ', ')); end
[hcols, sortIdx] = sort(hnum(valid));
colNamesSorted = vnames(valid); colNamesSorted = colNamesSorted(sortIdx);

M = W{:, colNamesSorted};                      % scores matrix

% ---------- Order models by overall mean (BEST FIRST) ----------
overallMean = mean(M, 2, 'omitnan');
[~, order] = sort(overallMean, 'descend');     % best first
M = M(order, :);
modelNames = modelNames(order);

% ---------- Score -> Rank (optional, 1 = best) ----------
if strcmpi(opt.mode,'rank')
    R = nan(size(M));
    for j = 1:size(M,2)
        col = M(:,j);
        [~,~,r] = unique(-col, 'stable');      % higher score -> rank 1
        r(isnan(col)) = NaN;
        R(:,j) = r;
    end
    Z = R;
else
    Z = M;
end

% ---------- Figure ----------
fh = figure('Color','w','Units','normalized','Position',[0.12 0.12 0.62 0.66], 'Visible','on');
ax = axes(fh); hold(ax,'on'); box(ax,'on');
set(ax,'YDir','reverse');   % <<--- row 1 at the TOP (best models at top)

imagesc(ax, Z, 'AlphaData', ~isnan(Z));   % hide NaNs

% Colormap + colorbar
if strcmpi(opt.mode,'rank')
    colormap(ax, flipud(parula));         % bright for rank=1
    cb = colorbar; ylabel(cb,'Rank (1 = best)');
else
    colormap(ax, parula);
    cb = colorbar; ylabel(cb, metric);
end
if ~isempty(opt.clim), caxis(ax, opt.clim); end

% X tick labels
try
    xtlbl = compose('Month+%d', hcols);
catch
    xtlbl = arrayfun(@(k)sprintf('Month+%d',k), hcols, 'UniformOutput', false);
end
set(ax,'XTick',1:size(Z,2), 'XTickLabel', xtlbl, ...
       'YTick',1:size(Z,1), 'YTickLabel', modelNames, ...
       'TickLabelInterpreter','none', 'FontSize', opt.baseFontSize);

xlabel(ax,'Forecast Horizon', 'FontSize', opt.baseFontSize);
if strcmpi(opt.mode,'rank')
    ylabel(ax,'Models (rank: 1 = best)', 'FontSize', opt.baseFontSize);
else
    ylabel(ax, sprintf('%s — Models', metric), 'FontSize', opt.baseFontSize);
end

% Title
% Title
if strcmpi(opt.mode,'rank')
    base = 'Ranks';
else
    base = metric;   % metric is already a char
end
ttl = string(opt.titleStr);
if strlength(ttl)==0
    ttl = sprintf('%s per model × horizon — All datasets', base);
end
title(ax, ttl, 'FontWeight','bold', 'FontSize', opt.baseFontSize+2);


% ---------- Numeric annotations with auto-contrast ----------
cmap = colormap(ax); clim = get(ax,'CLim'); nC = size(cmap,1);
[numRows, numCols] = size(Z);
for i = 1:numRows
    for j = 1:numCols
        if ~isnan(Z(i,j))
            if strcmpi(opt.mode,'rank')
                txt = sprintf('%d', round(Z(i,j)));
            else
                txt = sprintf(sprintf('%%.%df', opt.precision), Z(i,j));
            end
            t = (Z(i,j) - clim(1)) / max(eps, (clim(2)-clim(1))); t = min(max(t,0),1);
            ci = max(1, min(nC, round(1 + t*(nC-1))));
            rgb = cmap(ci,:); lum = 0.2126*rgb(1) + 0.7152*rgb(2) + 0.0722*rgb(3);
            txtColor = 'k'; if lum < 0.5, txtColor = 'w'; end

            text(j,i,txt,'HorizontalAlignment','center','VerticalAlignment','middle', ...
                'Color',txtColor,'FontSize',max(8,opt.baseFontSize-1),'FontWeight','bold');
        end
    end
end
axis(ax,'tight');               % no 'axis ij' here

% ---------- Save ----------
outDir = char(opt.outDir); if isempty(outDir), outDir = pwd; end
[ok,msg] = mkdir(outDir); if ~ok, error('Could not create "%s": %s', outDir, msg); end
baseName = sprintf('heatmap_%s_all_datasets_%s', lower(metric), lower(opt.mode));
savefig(fh, fullfile(outDir, [baseName '.fig']));
exportgraphics(ax, fullfile(outDir, [baseName '.png']), 'Resolution', 300);

% Expose useful pieces
fh.UserData.metricMatrix = M;
fh.UserData.displayMatrix = Z;
fh.UserData.modelNames = modelNames;
fh.UserData.horizons = hcols;
fh.UserData.colorLimits = get(ax,'CLim');

end
