clear;
% =========================================================================
dataset = 'avenue';
model = 'AE-BN';
name = 'Avenue';
video_train = 1:16;
video_test = 1:21;
dataset_path = '/mnt/fastdataset/Datasets';
train_path = '../../data/avenue_train';
test_path = '../../data/avenue_test';
input_channels = 10;
sample_stride = 5;

% t-SNE
numDims = 2; pcaDims = 50; perplexity = 50; theta = 0.7; alg = 'svd';

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
z_data = [];
z_mse = [];
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
    [z_mse_cur, video_ids_cur, sample_ids_cur, is_mse_high_cur, ...
        is_mse_low_cur] = ReadMSE(train_path, dataset, model, ...
        video_id, sample_interval_train, mse_high_rank, mse_low_rank);

    z_mse = [z_mse; z_mse_cur];
    
    tracking_table = [tracking_table; ...
        [video_ids_cur, sample_ids_cur, zeros(length(video_ids_cur), 1)]]; 
    
    disp('  Load data is done');
end

z_train_mean = mean(z_mse);
z_train_std = std(z_mse);
z_train_max = max(z_mse);
z_train_high_mean = max(z_mse(z_mse > 1.5 * z_train_std + z_train_mean));

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
z_test_mse = [];
for video_id = video_test
    
    counter = counter + 1;
    fprintf('[%02d/%02d] %s video %02d', ...
        counter, num_videos, dataset, video_id);    
    
    % read latent variables
    [z_mse_cur, video_ids_cur, sample_ids_cur, is_mse_high_cur, ...
        is_mse_low_cur, num_z] = ReadMSE(test_path, dataset, model, ...
        video_id, sample_interval_test, mse_high_rank, mse_low_rank);        
    z_test_mse = [z_test_mse; z_mse_cur];
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

z_normal_mse = z_test_mse(~is_positive);
z_normal_mean = mean(z_normal_mse);
z_normal_std = std(z_normal_mse);
z_normal_max = max(z_normal_mse);
z_normal_high_mean = max(z_normal_mse(z_normal_mse > 1.5 * z_normal_std + z_normal_mean));

z_abnormal_mse = z_test_mse(is_positive);
z_abnormal_mean = mean(z_abnormal_mse);
z_abnormal_std = std(z_abnormal_mse);
z_abnormal_max = max(z_abnormal_mse);
z_abnormal_high_mean = max(z_abnormal_mse(z_abnormal_mse > 1.5 * z_abnormal_std + z_abnormal_mean));


fig = figure; clf;
set(fig, 'position', [0, 100, 600, 400]);
h=histogram(z_mse);
title(sprintf('%s: train\nmean=%.3f, std=%.3f, \nmax=%.3f, high mean=%.3f', ...
    name, z_train_mean, z_train_std, z_train_max, z_train_high_mean));
%h.Normalization = 'countdensity';
h.BinWidth = 0.25;
h.NumBins = 200;
saveas(fig, '../../data/train.png')

fig = figure; clf;
set(fig, 'position', [600, 100, 600, 400]);
h=histogram(z_test_mse(is_positive));
title(sprintf('%s: abnormal\nmean=%.3f, std=%.3f, \nmax=%.3f, high mean=%.3f', ...
    name, z_abnormal_mean, z_abnormal_std, z_abnormal_max, z_abnormal_high_mean));
%h.Normalization = 'countdensity';
h.BinWidth = 0.25;
h.NumBins = 200;
saveas(fig, '../../data/abnormal.png')

fig = figure; clf;
set(fig, 'position', [1200, 100, 600, 400]);
h=histogram(z_test_mse(~is_positive));
title(sprintf('%s: normal\nmean=%.3f, std=%.3f, \nmax=%.3f, high mean=%.3f', ...
    name, z_normal_mean, z_normal_std, z_normal_max, z_normal_high_mean));
%h.Normalization = 'countdensity';
h.BinWidth = 0.25;
h.NumBins = 200;
saveas(fig, '../../data/normal.png')


decision_threshold = 0:0.1:1000;
precision = zeros(1, length(decision_threshold));
recall = zeros(1, length(decision_threshold));
i = 1;
for th = decision_threshold
    
    FP = sum(z_normal_mse > th);
    TN = sum(z_normal_mse <= th);
    FN = sum(z_abnormal_mse < th);
    TP = sum(z_abnormal_mse >= th);
    
    precision(i) = TP / (TP + FP);
    recall(i)    = TP / (TP + FN);
    
    i = i + 1;
end

figure; clf;
hold on;
grid on;
plot(recall, precision, '-r');
xlabel('recall');
ylabel('precision');
title('ROC curve');
hold off;


figure; clf;
hold on;
grid on;
plot(decision_threshold, precision, '-r');
plot(decision_threshold, recall, '-b');
xlabel('decision threshold');
ylabel('value');
title('Precision and recall on MSE threshold');
legend('precision', 'recall');
hold off;
