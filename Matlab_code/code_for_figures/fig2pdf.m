function fig2pdf(figFile, pdfFile)
% fig2pdf Convert a .fig file to a .pdf file (vector, high quality)
%
% Usage:
%   fig2pdf('algorithm_analysis_main.fig', 'output.pdf')

    if nargin < 2
        error('Usage: fig2pdf(''input.fig'',''output.pdf'')');
    end

    % Open the .fig file
    h = openfig(figFile,'new','visible');

    % Make sure it's not docked
    set(h,'WindowStyle','normal');

    % Export as vector PDF (best for papers)
    exportgraphics(h, pdfFile, 'ContentType','vector');

    % Close the figure (optional)
    close(h);

    fprintf('Saved PDF: %s\n', pdfFile);
end
