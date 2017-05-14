% =========================================================================
sample_stride = 5;
dataset = 'avenue';
video = 1:21;
model = 'VAE-NARROW';
result_path = './data/avenue_test';
size_z = 200;
input_channels = 10;
% =========================================================================


%--------------------------------------
% data preparing
%--------------------------------------
% load ref data
load(fullfile('./data/avenue_train', 'z_info.mat'));
z_cov_inv = inv(z_cov);

% load ground truth
load(fullfile('ground_truth', sprintf('gt_%s.mat', dataset)));


%--------------------------------------
% compute distance
%--------------------------------------
% training sample distance
z_delta_train = z_data - repmat(z_mean, size(z_data, 1), 1);
mahal_dist_train = ...
    sqrt(sum(z_delta_train * z_cov_inv .* z_delta_train, 2));
disp('Compute distances of positive data is done');

num_videos = length(video);
counter = 0;
mahal_dist = [];
pos_portion = [];
for video_idx = video
    
    counter = counter + 1;
    fprintf('[%02d/%02d] %s video %02d', ...
        counter, num_videos, dataset, video_idx);
    
    % load z
    file_name = sprintf('%s_video_%02d_%s_latent_variables.txt', ...
            dataset, video_idx, model);
    filepath = fullfile(result_path, file_name);
    read_data = csvread(filepath);
    z_video = read_data(:,1:size_z);
    num_z_video = size(z_video, 1);
    disp('  Load data is done')
    
    % compute distance    
    z_delta = z_video - repmat(z_mean, num_z_video, 1);
    mahal_dist_cur = sqrt(sum(z_delta * z_cov_inv .* z_delta, 2));
    disp('  Compute distances of target data is done');
    mahal_dist = [mahal_dist; mahal_dist_cur];

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

%--------------------------------------
% draw distributions
%--------------------------------------
% density with kernel smoothing
[density_train, x_train] = ksdensity(mahal_dist_train);
[density_pos, x_pos] = ksdensity(mahal_dist(is_positive));
[density_neg, x_neg] = ksdensity(mahal_dist(~is_positive));

% distribution
figure(1); clf;
plot(x_train, density_train, '-k');
hold on;
plot(x_pos, density_pos, '-r');
plot(x_neg, density_neg, '-b');
grid on;
title(sprintf('distance distribution: %s', dataset));
xlabel('distance');
ylabel('density');
legend('train', 'pos', 'neg');
hold off;

% population
figure(2); clf;
plot(x_train, density_train * size(mahal_dist_train, 1), ':k');
hold on;
plot(x_pos, density_pos * sum(find(is_positive)), '-r');
plot(x_neg, density_neg * sum(find(~is_positive)), '-b');
grid on;
title(sprintf('distance distribution: %s', dataset));
xlabel('distance');
ylabel('density');
legend('train', 'pos', 'neg');
hold off;

% figure(2); clf;
% plot(x_pos, density_pos * sum(find(is_positive)), '-r');
% hold on;
% plot(x_neg, density_neg * sum(find(~is_positive)), '-b');
% grid on;
% title(sprintf('distance population: %s', dataset));
% xlabel('distance');
% ylabel('population');
% legend('pos', 'neg');
% hold off;