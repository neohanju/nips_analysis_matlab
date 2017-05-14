clear
% =========================================================================
sample_stride = 5;
dataset = 'avenue';
video_train = 1:16;
video_test = 1:21;
model = 'VAE-NARROW';
train_path = '../../data/avenue_train';
test_path = '../../data/avenue_test';
size_z = 200;
input_channels = 10;
numDims = 2; pcaDims = 50; perplexity = 50; theta = 0.7; alg = 'svd';
tr_interval_half = 1;  % transition interval
sample_interval_train = 10;
sample_interval_test = 2;
% =========================================================================


%--------------------------------------
% data preparing
%--------------------------------------
z_data = zeros(0, size_z);
cnt = 0;
for video_id = video_train
    cnt = cnt + 1;
    fprintf('%d/%d\n', cnt, length(video_train));
    % read latent variables
    file_name = sprintf('%s_video_%02d_%s_latent_variables.txt', ...
        dataset, video_id, model);
    filepath = fullfile(train_path, file_name);
    read_data = importdata(filepath);
    z_cur = read_data(:,1:size_z);    
    z_data = [z_data; z_cur];    
end
z_data = z_data(1:sample_interval_train:end,:);
labels = zeros(size(z_data, 1), 1);

% load ground truth
load(fullfile('../../ground_truth', sprintf('gt_%s.mat', dataset)));

%--------------------------------------
% pos/neg
%--------------------------------------
num_videos = length(video_test);
counter = 0;
pos_portion = [];
z_test = zeros(0, size_z);
for video_idx = video_test
    
    counter = counter + 1;
    fprintf('[%02d/%02d] %s video %02d', ...
        counter, num_videos, dataset, video_idx);
    
    % load z
    file_name = sprintf('%s_video_%02d_%s_latent_variables.txt', ...
            dataset, video_idx, model);
    filepath = fullfile(test_path, file_name);
    z_cur = importdata(filepath);
    z_cur = z_cur(1:sample_interval_test:end,:);
    num_z_video = size(z_cur, 1);
    z_test = [z_test; z_cur];
    disp('  Load data is done');

    % pos/neg
    gt_interval = gt{video_idx};
    num_frames = (num_z_video - 1) * sample_stride + input_channels;
    gt_indicators = zeros(1, num_frames);
    for c = 1:size(gt_interval, 2)
        gt_indicators(gt_interval(1,c):gt_interval(2,c)) = 1;
    end
    pos_portion_cur = zeros(1, num_z_video);    
    for i = 1:num_z_video
        w_start = (i -1) * sample_stride + 1;
        w_end   = w_start + input_channels - 1;
        pos_portion_cur(i) = ...
            1/input_channels * sum(gt_indicators(w_start:w_end));
    end    
    pos_portion = [pos_portion, pos_portion_cur];
end
is_positive = false(size(pos_portion));
is_positive(pos_portion > 0) = true;

% --------------------------------------
% remove transition region
% --------------------------------------
is_transition = false(size(pos_portion));
for sample_idx = 2:length(pos_portion)
    if abs(pos_portion(sample_idx) - pos_portion(sample_idx-1)) ~= 0
        start_pos = max(1,sample_idx-tr_interval_half);
        end_pos   = min(sample_idx+tr_interval_half-1,length(pos_portion));
        is_transition(start_pos:end_pos) = true;
    end
end
% count deleted abnormal events
num_abnormal = 0;
num_left_abnormal = 0;
is_left = is_positive & ~is_transition;
b_positive = false;
b_left = false;
start_pos = 1;
end_pos = 1;
for pos = 1:length(is_positive)
    if ~b_positive && is_positive(pos)
        b_positive = true;
        start_pos = pos;
        num_abnormal = num_abnormal + 1;
    elseif b_positive && (~is_positive(pos) || pos == length(is_positive))
        b_positive = false;
        end_pos = pos - 1 * (~is_positive(pos));
        if sum(is_left(start_pos:end_pos)) > 0
            num_left_abnormal = num_left_abnormal + 1;
        end
    end
end
fprintf('Total %d abnoramls, %d are removed\n', ...
    num_abnormal, num_abnormal - num_left_abnormal);

remove_ratio = sum(is_transition) / length(is_transition) * 100;
fprintf('Removing %.2f %% of test samples as transition samples\n', ...
    remove_ratio);

z_test(is_transition,:) = [];
is_positive(is_transition) = [];

%--------------------------------------
% t-SNE
%--------------------------------------
z_data = [z_data; z_test];

labels_test = ones(length(is_positive), 1);
labels_test(is_positive) = 2;
labels = [labels; labels_test];
disp('Labeling is done! All data are prepared');
disp('Do t-sne...');

map = fast_tsne(z_data, numDims, pcaDims, perplexity, theta, alg);

disp('Done!');

figure(10); clf;
grid on;
hold on;
if numDims == 2
    gscatter(map(:,1), map(:,2), labels, 'bgr', '+ox', 5);
elseif numDims == 3
    cmap = colormap('parula');
    scatter3(map(labels == 0,1), map(labels == 0,2), map(labels == 0,3), 20, cmap(1,:));
    scatter3(map(labels == 1,1), map(labels == 1,2), map(labels == 1,3), 20, cmap(30,:));
    scatter3(map(labels == 2,1), map(labels == 2,2), map(labels == 2,3), 20, cmap(60,:));
end
title(sprintf('PCA=%d,perplexity=%d,theta=%f', pcaDims, perplexity, theta));
legend('train', 'normal', 'abnormal');
hold off;
