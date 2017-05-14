clear

% =========================================================================
sample_stride = 5;
dataset = 'avenue';
video = 1:21;
model = 'VAE-NARROW';
result_path = './data/avenue_test';
size_z = 200;
input_channels = 10;
% =========================================================================

% graph config
fig_horizontal_num = 5;
fig_vertical_num = 4;
fig_w = 384;
fig_h = 270;

% load ground truth
load(fullfile('ground_truth', sprintf('gt_%s.mat', dataset)));

cnt = 0;
for video_id = video
    cnt = cnt + 1;
    fprintf('%d/%d\n', cnt, length(video));

    % read latent variables
    file_name = sprintf('%s_video_%02d_%s_latent_variables.txt', ...
        dataset, video_id, model);    
    read_data = importdata(fullfile(result_path, file_name));
    matZ = read_data(:,1:200)';

    [numZ, numF] = size(matZ);
    [Fs, Zs] = meshgrid(sample_stride:sample_stride:sample_stride*numF, ...
        1:1:numZ);

    % produce ground truth mat
    gt_interval = gt{video_id};
    label_intesity = max(max(matZ));
    matGT = zeros(numF*sample_stride, numZ);
    for c = 1:size(gt_interval, 2)
        start_pos = gt_interval(1,c);
        end_pos = gt_interval(2,c);
        matGT(start_pos:end_pos,:) = label_intesity;
    end
    surface_offset = 15;
    GT_surface = zeros(size(matGT)) - surface_offset;

    % draw figures
    fig = figure(cnt); clf;    
    colormap('jet');
    h = waterfall(Fs, Zs, matZ);
    set(h, 'LineWidth', 4);    
    CD = get(h, 'CData');
    CD(1,:) = nan;
    CD(end-2:end,:) = nan;
    set(h, 'CData', CD)
    hidden off;
    grid on;
    title(sprintf('%s video %d', dataset, video_id));
    xlabel('time');
    ylabel('axis');
    zlabel('z');

    hold on;
    % ground truth
    surface(GT_surface', matGT', 'FaceColor', 'texturemap', ...
        'EdgeColor', 'none', 'CDataMapping', 'direct')
    hold off;
    view([-4 31]);
    drawnow;
    y_pos = fig_h*(floor((cnt-1)/fig_horizontal_num));
    if cnt > fig_horizontal_num * fig_vertical_num
        y_pos = y_pos - 1920;
    end
    set(fig, 'menubar', 'none', 'position', ...
        [rem(cnt-1, fig_horizontal_num)*fig_w, ...
         y_pos, ...
         fig_w, fig_h]);
end

%()()
%('')HAANJU.YOO