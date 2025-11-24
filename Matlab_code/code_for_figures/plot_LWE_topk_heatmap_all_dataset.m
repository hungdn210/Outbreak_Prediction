function fh = plot_LWE_topk_heatmap_all_dataset(csvFile, varargin)
% plot_LWE_topk_heatmap_all_dataset
% Heatmap of LWE ensemble performance:
%   rows   = baseline LWE + top-k ensemble sizes
%   cols   = forecast horizons (Month+1 ... Month+K)
%   values = mean(metric) across ALL datasets
%
% Example:
%   fh = plot_LWE_topk_heatmap_all_dataset( ...
%            'LWE_comparison.csv', ...
%            'metric','F2', ...
%            'outDir','figures', ...
%            'titleStr','LWE ensemble size vs. forecast horizon', ...
%            'baseFontSize',12);

% ------------- parse args -------------
p = inputParser;
addParameter(p,'metric','F2',@(s)ischar(s)||isstring(s));
addParameter(p,'outDir','figures',@(s)ischar(s)||isstring(s));
addParameter(p,'titleStr','',@(s)ischar(s)||isstring(s));
addParameter(p,'baseFontSize',11,@isscalar);
parse(p,varargin{:});
opt    = p.Results;
metric = char(opt.metric);

% ------------- read table -------------
T = readtable(csvFile,'TextType','string');

% keep only LWE variants
T = T(contains(T.model,"LWE"), :);
if isempty(T)
    error('No rows with model containing "LWE" found in %s', csvFile);
end

% ---- derive horizon from "Month+1"... ----
mh = regexp(string(T.month),'Month\+(\d+)','tokens','once');
T.horizon = nan(height(T),1);
for i = 1:height(T)
    tok = mh{i};
    if ~isempty(tok)
        T.horizon(i) = str2double(tok{1});
    end
end
T = T(~isnan(T.horizon), :);

% ---- ensure metric numeric ----
if ~isnumeric(T.(metric))
    T.(metric) = str2double(string(T.(metric)));
end
T = T(~isnan(T.(metric)), :);

% ---- parse ensemble size K from model name ----
% "LWE - top 9 models" -> ensK = 9
% "LWE" (baseline)     -> ensK = 0
modelStr = string(T.model);
tokens = regexp(modelStr, 'top\s+(\d+)\s+models', 'tokens','once');

ensK = zeros(height(T),1);   % default 0 for baseline LWE
for i = 1:height(T)
    tk = tokens{i};
    if ~isempty(tk)
        ensK(i) = str2double(tk{1});
    end
end
T.ensK = ensK;

% ---- aggregate: mean metric per (ensK, horizon) over ALL datasets ----
G = groupsummary(T, {'ensK','horizon'}, 'mean', metric);
G.Properties.VariableNames{end} = 'MeanMetric';

% sort ensemble sizes and horizons nicely
Klist = unique(G.ensK);
Klist = sort(Klist,'ascend');        % 0 (baseline) first
Hlist = unique(G.horizon);
Hlist = sort(Hlist,'ascend');

% ---- build matrix for heatmap ----
M = nan(numel(Klist), numel(Hlist));
for i = 1:numel(Klist)
    k = Klist(i);
    subK = G(G.ensK==k,:);
    for j = 1:numel(Hlist)
        h = Hlist(j);
        row = subK(subK.horizon==h,:);
        if ~isempty(row)
            M(i,j) = row.MeanMetric;
        end
    end
end

% ---- plot heatmap ----
fh = figure('Color','w','Units','normalized', ...
            'Position',[0.2 0.15 0.55 0.6],'Visible','on');
ax = axes(fh); 
imagesc(Hlist, 1:numel(Klist), M);
set(ax,'YDir','normal');     % row 1 at top
colormap(parula);
colorbar;
grid(ax,'on');
ax.Layer = 'top';

% y-tick labels: baseline + "Top-k"
yticks(1:numel(Klist));
yLabels = strings(numel(Klist),1);
for i = 1:numel(Klist)
    if Klist(i)==0
        yLabels(i) = "LWE (baseline)";
    else
        yLabels(i) = sprintf('Top-%d models', Klist(i));
    end
end
yticklabels(yLabels);

xlabel(ax,'Forecast horizon (Month +k)', ...
       'FontSize',opt.baseFontSize);
ylabel(ax,'Ensemble configuration', ...
       'FontSize',opt.baseFontSize);

ttl = opt.titleStr;
if isempty(ttl)
    ttl = sprintf('LWE ensemble size vs. horizon (%s, all datasets)', metric);
end
title(ax, ttl, 'FontSize', opt.baseFontSize+2, 'FontWeight','bold');

set(ax,'FontSize',opt.baseFontSize);

% ---- annotate cells with metric values ----
for i = 1:numel(Klist)
    for j = 1:numel(Hlist)
        if ~isnan(M(i,j))
            text(Hlist(j), i, sprintf('%.3f', M(i,j)), ...
                'HorizontalAlignment','center', ...
                'VerticalAlignment','middle', ...
                'Color','k', 'FontSize', opt.baseFontSize-1);
        end
    end
end

% ---- save FIG ----
outDir = char(opt.outDir);
if isempty(outDir), outDir = pwd; end
[ok,msg] = mkdir(outDir);
if ~ok, error('Could not create outDir "%s": %s', outDir, msg); end

baseName = sprintf('LWE_topk_heatmap_%s_all_datasets', lower(metric));
outFIG   = fullfile(outDir, [baseName '.fig']);
savefig(fh, outFIG);

fh.UserData.outputPathFIG = outFIG;
fh.UserData.matrix = M;
fh.UserData.Klist = Klist;
fh.UserData.Hlist = Hlist;

end
