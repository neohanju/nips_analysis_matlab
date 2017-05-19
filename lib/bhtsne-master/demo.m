clear;
% =========================================================================
dataset = 'avenue';
video_train = 1:16;
video_test = 1:21;
model = 'AE-BN';
dataset_path = '/mnt/fastdataset/Datasets';
train_path = '../../data/avenue_train';
test_path = '../../data/avenue_test';
input_channels = 10;
sample_stride = 5;

% t-SNE
numDims = 2; pcaDims = 50; perplexity = 50; theta = 0.3; alg = 'svd';

% sample selection
tr_interval_half = 1;  % transition interval
sample_interval_train = 1;
sample_interval_test = 1;
mse_high_rank = 1;
mse_low_rank = 1;
% =========================================================================


%--------------------------------------
% data preparing
%--------------------------------------
disp('Preparing train data');
z_train = [];
z_train_mse = [];
is_mse_high = false(0);
is_mse_low = false(0);
video_ids = [];
sample_ids = [];
num_videos = length(video_train);
tracking_table = [];
counter = 0;
for video_id = video_train
    counter = counter + 1;
    fprintf('[%02d/%02d] %s video %02d', ...
        counter, num_videos, dataset, video_id);
    
    % read latent variables
    [z_cur, z_mse_cur, video_ids_cur, sample_ids_cur, is_mse_high_cur, ...
        is_mse_low_cur] = ReadSamples(train_path, dataset, model, ...
        video_id, sample_interval_train, mse_high_rank, mse_low_rank);

    z_train = [z_train; z_cur];
    z_train_mse = [z_train_mse; z_mse_cur];
    is_mse_high = [is_mse_high; is_mse_high_cur];
    is_mse_low = [is_mse_low; is_mse_low_cur];   
    tracking_table = [tracking_table; ...
        [video_ids_cur, sample_ids_cur, zeros(length(video_ids_cur), 1)]];
    disp('  Load data is done');
end
labels = zeros(length(is_mse_high), 1);
labels(is_mse_low) = 3;
labels(is_mse_high) = 4;

% load ground truth
load(fullfile('../../ground_truth', sprintf('gt_%s.mat', dataset)));


%--------------------------------------
% pos/neg
%--------------------------------------
disp('Preparing test data');
num_videos = length(video_test);
counter = 0;
pos_portion = [];
num_original_abnormal = 0;
z_test = [];
z_test_mse = [];
test_is_mse_high = false(0);
test_is_mse_low = false(0);
for video_id = video_test
    
    counter = counter + 1;
    fprintf('[%02d/%02d] %s video %02d', ...
        counter, num_videos, dataset, video_id);    
    
    % read latent variables
    [z_cur, z_mse_cur, video_ids_cur, sample_ids_cur, is_mse_high_cur, ...
        is_mse_low_cur, num_z] = ReadSamples(test_path, dataset, model, ...
        video_id, sample_interval_test, mse_high_rank, mse_low_rank);    
    z_test = [z_test; z_cur];
    z_test_mse = [z_test_mse; z_mse_cur];
    test_is_mse_high = [test_is_mse_high; is_mse_high_cur];
    test_is_mse_low = [test_is_mse_low; is_mse_low_cur];
    tracking_table = [tracking_table; ...
        [video_ids_cur, sample_ids_cur, ones(length(video_ids_cur), 1)]];    
    disp('  Load data is done');

    % pos/neg
    gt_interval = gt{video_id};
    num_frames = (num_z - 1) * sample_stride + input_channels;
    gt_indicators = zeros(1, num_frames);
    num_original_abnormal = num_original_abnormal + size(gt_interval, 2);
    for c = 1:size(gt_interval, 2)
        gt_indicators(gt_interval(1,c):gt_interval(2,c)) = 1;
    end
    pos_portion_cur = zeros(1, num_z);    
    for i = 1:num_z
        w_start = (i -1) * sample_stride + 1;
        w_end   = w_start + input_channels - 1;
        pos_portion_cur(i) = ...
            1/input_channels * sum(gt_indicators(w_start:w_end));
    end    
    pos_portion = [pos_portion, pos_portion_cur(sample_ids_cur)];
end
is_positive = false(size(pos_portion));
is_positive(pos_portion > 0) = true;

labels_test = ones(length(is_positive), 1);
labels_test(is_positive) = 2;
labels_test(test_is_mse_low) = 5;
labels_test(test_is_mse_high) = 6;


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
fprintf('Origianlly %d anomalies\n', num_original_abnormal);
fprintf('Total %d anomalies are sampled and %d are removed\n', ...
    num_abnormal, num_abnormal - num_left_abnormal);

remove_ratio = sum(is_transition) / length(is_transition) * 100;
fprintf('Removing %.2f %% of test samples as transition samples\n', ...
    remove_ratio);

z_test(is_transition,:) = [];
z_test_mse(is_transition) = [];
is_positive(is_transition) = [];


%--------------------------------------
% t-SNE
%--------------------------------------
z_data = [z_train; z_test];

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
% 0=train / 1=test_noraml / 2=test_abnormal / 3=low_mse / 4=high_mse
if numDims == 2
    gscatter(map(:,1), map(:,2), labels, 'kbmcryg', '+ox^svd', 5);
elseif numDims == 3
    cmap = colormap('parula');
    scatter3(map(labels == 0,1), map(labels == 0,2), map(labels == 0,3), 20, cmap(1,:));
    scatter3(map(labels == 1,1), map(labels == 1,2), map(labels == 1,3), 20, cmap(30,:));
    scatter3(map(labels == 2,1), map(labels == 2,2), map(labels == 2,3), 20, cmap(60,:));
end
title(sprintf('PCA=%d,perplexity=%d,theta=%f', pcaDims, perplexity, theta));
legend('train', 'normal', 'abnormal', 'train mse low', 'train mse high', ...
    'test mse low', 'test mse high');
hold off;

% show sample example
% ShowSample(1881, dataset, dataset_path, tracking_table);

% scatter with MSE
num_color = 300;
cmap = colormap(jet(num_color));

z_train_mse = z_mse;
z_mse = [z_train_mse; z_test_mse];
z_mes_ = (z_mse - min(z_mse))/(max(z_mse) - min(z_mse) + 1);
color_idx = floor(z_mes_ * num_color + 1);

train_sample_idx = 1:length(z_train_mse);
test_sample_idx = length(z_train_mse):length(z_mse);
abnormal_sample_idx = test_sample_idx(is_positive);
normal_sample_idx = test_sample_idx(~is_positive);

figure(20); clf;
hold on;
target_idx = train_sample_idx;
% scatter(map(target_idx,1), map(target_idx,2), 40, ...
%     cmap(color_idx(target_idx),:), 'filled', ...
%     'MarkerEdgeColor', [0 0 1], ...
%     'LineWidth', 1);
% target_idx = normal_sample_idx;
% scatter(map(target_idx,1), map(target_idx,2), 40, ...
%     cmap(color_idx(target_idx),:), 'filled', ...
%     'MarkerEdgeColor', [0 0.7 0], ...
%     'LineWidth', 1);
% target_idx = abnormal_sample_idx;
% scatter(map(target_idx,1), map(target_idx,2), 40, ...
%     cmap(color_idx(target_idx),:), 'filled', ...
%     'MarkerEdgeColor', [1 0 0], ...
%     'LineWidth', 1);
scatter(map(target_idx,1), map(target_idx,2), 50, ...
    cmap(color_idx(target_idx),:), 'filled', ...    
    'LineWidth', 1);
target_idx = normal_sample_idx;
scatter(map(target_idx,1), map(target_idx,2), 50, ...
    cmap(color_idx(target_idx),:), 'filled', ...
    'LineWidth', 1);
target_idx = abnormal_sample_idx;
scatter(map(target_idx,1), map(target_idx,2), 50, ...
    cmap(color_idx(target_idx),:), 'filled', ...
    'MarkerEdgeColor', [1 0 0], ...
    'LineWidth', 2);
hold off;

num_tick_labels = 11;
z_mse_range = max(z_mse) - min(z_mse);
colorbar_label = cell(length(num_tick_labels), 1);
cur_tick_mse = min(z_mse);
for i = 1:num_tick_labels    
    colorbar_label{i} = sprintf('%.2f', cur_tick_mse);
    cur_tick_mse = cur_tick_mse + round(z_mse_range/num_tick_labels);a
end
colorbar('Location', 'EastOutside', 'YTickLabel', colorbar_label)


%--------------------------------------
% Entropy
%--------------------------------------
mse_sum = sum(z_train_mse);
P_z_train_mse = z_train_mse / mse_sum;
H_train = sum(P_z_train_mse .* log(P_z_train_mse));


%()()
%('')HAANJU.YOO
