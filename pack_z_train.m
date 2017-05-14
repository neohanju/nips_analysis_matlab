clear

% =========================================================================
sample_stride = 5;
dataset = 'avenue';
video = 1:16;
model = 'VAE-NARROW';
result_path = './data/avenue_train';
size_z = 200;
% =========================================================================

z_data = zeros(0, size_z);
z_vel = zeros(0, size_z);

cnt = 0;
for video_id = video
    cnt = cnt + 1;
    fprintf('%d/%d\n', cnt, length(video));
    % read latent variables
    file_name = sprintf('%s_video_%02d_%s_latent_variables.txt', ...
        dataset, video_id, model);
    filepath = fullfile(result_path, file_name);
    read_data = csvread(filepath);
    z_cur = read_data(:,1:size_z);    
    num_z = size(z_cur, 1);
    z_cur(floor(num_z*0.5):end,:) = [];    
    z_data = [z_data; z_cur];    
end

save('z_data_stride_1.mat', 'z_data');