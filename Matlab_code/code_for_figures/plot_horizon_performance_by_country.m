function fh = plot_horizon_performance_by_country(csvFile, varargin)
% plot_horizon_performance
% Draws "Model Performance Across Horizons" line plots.
%
% Example:
%   fh = plot_horizon_performance('results_log_main.csv', ...
%        'metric','F2','data_id','France_level_2_final','outDir','figures')

% ---- Parse args ----
p = inputParser;
addParameter(p, 'metric', 'F2', @(s)ischar(s) || isstring(s));
addParameter(p, 'panelBy', 'country', @(s)ischar(s) || isstring(s));
addParameter(p, 'country', '', @(s)ischar(s) || isstring(s));
addParameter(p, 'data_id', '', @(s)ischar(s) || isstring(s));   % NEW
addParameter(p, 'modelName', '', @(s)ischar(s) || isstring(s));
addParameter(p, 'outDir', 'figures', @(s)ischar(s) || isstring(s));
addParameter(p, 'titleStr', '', @(s)ischar(s) || isstring(s));
addParameter(p, 'lineWidth', 1.8, @isscalar);
addParameter(p, 'baseFontSize', 11, @isscalar);
parse(p, varargin{:});
opt = p.Results;

metric     = char(opt.metric);
panelBy    = lower(char(opt.panelBy));
onlyCountry= strtrim(string(opt.country));
onlyModel  = strtrim(string(opt.modelName));
onlyDataID = strtrim(string(opt.data_id));    % NEW

% ---- Read CSV ----
T = readtable(csvFile, 'TextType', 'string');

% Guard: required columns
reqCols = ["data_id","model","month",metric];
missing = setdiff(reqCols, string(T.Properties.VariableNames));
if ~isempty(missing)
    error('Missing required columns in CSV: %s', strjoin(missing, ', '));
end

% ---- Derived columns ----
T.country = extractBefore(T.data_id, "_");
T.country(T.country=="") = T.data_id(T.country==""); % fallback

% Horizon number from 'Month+1'...'Month+6'
mh = regexp(string(T.month),'Month\+(\d+)','tokens','once');
T.horizon = nan(height(T),1);
for i = 1:height(T)
    tok = mh{i};
    if ~isempty(tok); T.horizon(i) = str2double(tok{1}); end
end
T = T(~isnan(T.horizon), :);

% Ensure metric is numeric
if ~isnumeric(T.(metric)), T.(metric) = str2double(string(T.(metric))); end
T = T(~isnan(T.(metric)), :);

% ---- Optional filtering ----
if ~strcmp(onlyDataID,"")                       % NEW: filter by exact data_id
    T = T(T.data_id == onlyDataID, :);
elseif strcmp(panelBy,'country') && ~strcmp(onlyCountry,"")
    T = T(T.country == onlyCountry, :);
elseif strcmp(panelBy,'model') && ~strcmp(onlyModel,"")
    T = T(T.model == onlyModel, :);
end
if isempty(T); error('No rows after filtering. Check country/model/data_id and CSV contents.'); end

% Sort for consistent plotting
T = sortrows(T, {'country','model','horizon'});

% ---- Determine panels & line keys ----
if ~strcmp(onlyDataID,"")                       % NEW: single exact dataset
    panels   = string(onlyDataID);
    lineKeys = unique(T.model, 'stable');
    panelLabel = 'Dataset';
elseif strcmp(panelBy,'country')
    panels   = unique(T.country, 'stable');
    lineKeys = unique(T.model, 'stable');      % lines = models
    panelLabel = 'Country';
elseif strcmp(panelBy,'model')
    panels   = unique(T.model, 'stable');
    lineKeys = unique(T.country, 'stable');    % lines = countries
    panelLabel = 'Model';
else
    error('panelBy must be ''country'' or ''model''.');
end

% ---- Figure ----
fh = figure('Color','w','Units','normalized','Position',[0.15 0.15 0.6 0.55],'Visible','on');
tlo = tiledlayout(fh, numel(panels), 1, 'TileSpacing','compact','Padding','compact');

titleStr = opt.titleStr;
if isempty(titleStr)
    if ~strcmp(onlyDataID,"")
        titleStr = sprintf('%s vs Horizon — %s', metric, onlyDataID);
    elseif strcmp(panelBy,'country') && ~strcmp(onlyCountry,"")
        titleStr = sprintf('%s vs Horizon — %s', metric, onlyCountry);
    elseif strcmp(panelBy,'model') && ~strcmp(onlyModel,"")
        titleStr = sprintf('%s vs Horizon — %s', metric, onlyModel);
    else
        titleStr = sprintf('%s vs Forecast Horizon', metric);
    end
end
sgtitle(tlo, titleStr, 'FontWeight','bold', 'FontSize', opt.baseFontSize+2);

% ---- For each panel ----
for pi = 1:numel(panels)
    ax = nexttile(tlo); hold(ax,'on'); grid(ax,'on'); box(ax,'on');
    panelVal = panels(pi);

    if ~strcmp(onlyDataID,"")
        S = T;  % already filtered
        xlab = 'Forecast Horizon (Month +k)';
    elseif strcmp(panelBy,'country')
        S = T(T.country==panelVal, :);
        xlab = 'Forecast Horizon (Month +k)';
    else
        S = T(T.model==panelVal, :);
        xlab = 'Forecast Horizon (Month +k)';
    end

    plotted = false;
    for lk = 1:numel(lineKeys)
        if strcmp(panelBy,'model')
            sub = S(S.country==lineKeys(lk), :);
        else
            sub = S(S.model==lineKeys(lk), :);
        end
        if isempty(sub); continue; end

        G = groupsummary(sub, 'horizon', 'mean', metric);
        horizons = (1:max(G.horizon))';
        M = nan(numel(horizons),1);
        [~,ia] = ismember(G.horizon, horizons);
        M(ia) = G.("mean_"+metric);

        key = char(lineKeys(lk));
        col = tableau20_colors(min(numel(lineKeys),20));
        col = col(lk,:);
        markers = {'o','s','^','v','d','>','<','p','h','x','+'};
        mk  = markers{mod(lk-1, numel(markers)) + 1};

        plot(horizons, M, '-', ...
            'Color', col, 'LineWidth', opt.lineWidth, ...
            'Marker', mk, 'MarkerFaceColor', col, 'MarkerSize', 5, ...
            'DisplayName', key);
        plotted = true;
    end

    if ~plotted
        text(ax,0.5,0.5,'No data','HorizontalAlignment','center');
    end

    title(ax, sprintf('%s: %s', panelLabel, panelVal), ...
        'FontWeight','bold','FontSize',opt.baseFontSize);
    xlabel(ax, xlab, 'FontSize', opt.baseFontSize);
    ylabel(ax, metric, 'FontSize', opt.baseFontSize);
    set(ax, 'FontSize', opt.baseFontSize, 'XLim',[0.8 6.2], 'XTick',1:6);

    leg = legend(ax, 'Location','southoutside','NumColumns',2); leg.Box = 'off';
end

% ---- Save FIG ----
if ~exist(opt.outDir, 'dir'), mkdir(opt.outDir); end
if ~strcmp(onlyDataID,"")
    baseName = sprintf('performance_over_horizon_%s_%s', lower(metric), onlyDataID);
else
    baseName = sprintf('performance_over_horizon_%s', lower(metric));
end
outFIG = fullfile(opt.outDir, baseName + ".fig");
savefig(fh, outFIG);

fh.UserData.outputPathFIG = outFIG;
fh.UserData.summary = G;

end


% =================== Local helper ===================
function C = tableau20_colors(n)
% Return the first n colors from Tableau 20 (distinct, publication-friendly).
base = [ ...
     31 119 180; 255 127  14;  44 160  44; 214  39  40; 148 103 189; ...
    140  86  75; 227 119 194; 127 127 127; 188 189  34;  23 190 207; ...
    174 199 232; 255 187 120; 152 223 138; 255 152 150; 197 176 213; ...
    196 156 148; 247 182 210; 199 199 199; 219 219 141; 158 218 229 ...
] / 255;
n = min(n, size(base,1));
C = base(1:n,:);
end
