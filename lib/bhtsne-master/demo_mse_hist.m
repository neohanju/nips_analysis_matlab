clear;
% =========================================================================
dataset = 'avenue';
model = 'AE-BN';
name = 'Avenue';
video_train = 1:16;
video_test = 1:21;
dataset_path = '/home/mlpa/Workspace/nips';
sample_path = '../../data/AE-BN_margin_2000';
train_path = fullfile(sample_path, '/avenue_train');
test_path = fullfile(sample_path, '/avenue_test');
input_channels = 10;
sample_stride = 5;

% sample selection
tr_interval_half = 1;  % transition interval
sample_interval_train = 1;
sample_interval_test = 1;
mse_high_rank = 1;
mse_low_rank = 1;

% test
test_mse_threshold = 400;

% drawing
b_draw_test_mse = false;
b_draw_train_mse = false;
b_draw_hist = false;
b_draw_ROC = false;
% =========================================================================

name_str = strrep(sample_path, '_', ' ');


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
    
    if b_draw_train_mse
        fig = figure; clf;
        grid on;
        hold on;
        title(sprintf('%s video %02d', name_str, video_id));
        plot(sample_ids_cur, z_mse_cur, '-');
        plot(sample_ids_cur, test_mse_threshold * ones(length(z_mse_cur), 1), '-g');        
        hold off;
        drawnow;
        saveas(fig, fullfile(sample_path, sprintf('train_result_%02d.png', video_id)))
    end
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
z_test_mse_per_video = cell(3, length(video_test));
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
    
    z_test_mse_per_video{1, counter} = z_mse_cur;
    z_test_mse_per_video{2, counter} = sample_ids_cur;

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
    
    if b_draw_test_mse
        fig = figure; clf;
        grid on;
        hold on;
        title(sprintf('%s video %02d', name_str, video_id));
        plot(sample_ids_cur, z_mse_cur, '-');
        plot(sample_ids_cur, test_mse_threshold * ones(length(z_mse_cur), 1), '-g');
        plot(sample_ids_cur, 2000 * pos_portion_cur, '-r');
        hold off;
        drawnow;
        saveas(fig, fullfile(sample_path, sprintf('test_result_%02d.png', video_id)))
    end
    
    z_test_mse_per_video{3, counter} = gt_indicators;
    
    filename = sprintf('avenue_video_%02d_conv3_iter_150000.txt', video_id);
    csvwrite(fullfile(test_path, filename), z_mse_cur);
    
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

if b_draw_hist
    fig = figure; clf;
    set(fig, 'position', [0, 100, 600, 400]);
    h=histogram(z_mse);
    title(sprintf('%s: train\nmean=%.3f, std=%.3f, \nmax=%.3f, high mean=%.3f', ...
        name, z_train_mean, z_train_std, z_train_max, z_train_high_mean));
    %h.Normalization = 'countdensity';
    h.BinWidth = 0.25;
    h.NumBins = 200;
    saveas(fig, fullfile(sample_path, 'train.png'))

    fig = figure; clf;
    set(fig, 'position', [600, 100, 600, 400]);
    h=histogram(z_test_mse(is_positive));
    title(sprintf('%s: abnormal\nmean=%.3f, std=%.3f, \nmax=%.3f, high mean=%.3f', ...
        name, z_abnormal_mean, z_abnormal_std, z_abnormal_max, z_abnormal_high_mean));
    %h.Normalization = 'countdensity';
    h.BinWidth = 0.25;
    h.NumBins = 200;
    saveas(fig, fullfile(sample_path, 'abnormal.png'))

    fig = figure; clf;
    set(fig, 'position', [1200, 100, 600, 400]);
    h=histogram(z_test_mse(~is_positive));
    title(sprintf('%s: normal\nmean=%.3f, std=%.3f, \nmax=%.3f, high mean=%.3f', ...
        name, z_normal_mean, z_normal_std, z_normal_max, z_normal_high_mean));
    %h.Normalization = 'countdensity';
    h.BinWidth = 0.25;
    h.NumBins = 200;
    saveas(fig, fullfile(sample_path, 'normal.png'))
end

decision_threshold = 0:50:1000;
% decision_threshold = 400;
precision = zeros(1, length(decision_threshold));
recall = zeros(1, length(decision_threshold));
frame_expand = sample_stride;
i = 1;
for th = decision_threshold
    
    TP = 0;
    FP = 0;
    FN = 0;
    
    for video_id = video_test
        num_samples = length(z_test_mse_per_video{1,video_id});
        num_frames = (num_samples-1)*sample_stride+input_channels;
        decision_positive = zeros(1, num_frames);
        for pos = 1:num_samples
            if z_test_mse_per_video{1,video_id}(pos) < th
                continue;
            end
            time_pos = (pos-1) * sample_stride + 1;
            start_pos = max(1, time_pos - frame_expand);
            end_pos = min(num_frames, time_pos + frame_expand);
            decision_positive(start_pos:end_pos) = 1;
        end
        
        bPositive = false;
        abnormal_regs = zeros(2, 0);
        for t = 1:num_frames
            if ~bPositive && decision_positive(t) == 1
                bPositive = true;
                abnormal_regs(:,end+1) = [t, 0]';
            end
            if bPositive && decision_positive(t) == 0
                bPositive = false;
                abnormal_regs(2,end) = t-1;
            end
        end
        
        % evaluate
		[det, gtg] = compute_overlaps(abnormal_regs, gt{video_id});
		TP = TP + sum(gtg==1);
		FP = FP + sum(det==0);
		FN = FN + sum(gtg==0);
        
        fig = figure(1000); clf;
        grid on;
        hold on;
        title(sprintf('%s video %02d', name_str, video_id));
        plot(1:num_frames, decision_positive + 0.5, '-');
%         plot([1:num_frames], th * ones(1, length(num_frames)), '-g');
        plot(1:num_frames, 2 * z_test_mse_per_video{3, video_id}, '-r');
        hold off;
        drawnow;        
        fprintf('[%02d] TP: %d FP: %d FN: %d\n', ...
            video_id, sum(gtg==1), sum(det==0), sum(gtg==0));
        pause;
    end
    
    precision(i) = TP / (TP + FP);
    recall(i)    = TP / (TP + FN);
    
    fprintf('Threshold: %.1f ', th);
    fprintf('\tTP: %d\t FP: %d\t FN: %d', TP, FP, FN);
    fprintf('\tPrecision: %0.2f ', precision(i));
    fprintf('\tRecall: %0.2f\n', recall(i));    
    
%     FP = sum(z_normal_mse > th);    
%     FN = sum(z_abnormal_mse < th);
%     TP = sum(z_abnormal_mse >= th);  
    
    i = i + 1;
end

if b_draw_ROC
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
end



%()()
%('')HAANJU.YOO
