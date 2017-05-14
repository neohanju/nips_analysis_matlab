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

save_path = fullfile('./data', 'recon_costs');

cnt = 0;
for video_id = video
    cnt = cnt + 1;
    fprintf('%d/%d\n', cnt, length(video));

    % read latent variables
    file_name = sprintf('%s_video_%02d_%s_latent_variables.txt', ...
        dataset, video_id, model);    
    read_data = importdata(fullfile(result_path, file_name));
    z_data = read_data(:,1:200)';

    % magnitude
    z_norm1 = sum(abs(z_data), 2);
    z_norm2 = sqrt(sum(z_data.^2, 2));
    
    % for competability, naming with recon_costs
    save_file_name = sprintf('%s_video_%02d_%s.txt', ...
        dataset, video_id, model);
    
    csvwrite(fullfile(save_path, save_file_name), z_norm2);
end

%()()
%('')HAANJU.YOO