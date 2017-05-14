function [z_data, z_mse, video_ids, sample_ids, is_mse_high, is_mse_low, ...
    original_num_z] = ...
        ReadSamples(path, dataset, model, video_id, ...
            sample_interval, mse_high_rank, mse_low_rank)

% read latent variables
file_name = sprintf('%s_video_%02d_%s_latent_variables.txt', ...
    dataset, video_id, model);
filepath = fullfile(path, file_name);
z_data = importdata(filepath);    
original_num_z = size(z_data, 1);

file_name = sprintf('%s_video_%02d_%s.txt', ...
    dataset, video_id, model);
filepath = fullfile(path, file_name);
z_mse = importdata(filepath);    

% sample selection
selected_samples = 1:sample_interval:original_num_z;  
z_data = z_data(selected_samples, :);
z_mse = z_mse(selected_samples, :);    

% find high/low mse samples
[~, idx] = sort(z_mse, 'ascend');
is_mse_high = false(length(idx), 1);
is_mse_low  = false(length(idx), 1);
is_mse_low(idx(1:mse_low_rank)) = true;
is_mse_high(idx(end:-1:end-mse_high_rank+1)) = true;
is_mse_high(is_mse_low) = false;  % high priority at low mse

% sample tracking    
video_ids = video_id*ones(length(selected_samples), 1);
sample_ids = selected_samples;


end