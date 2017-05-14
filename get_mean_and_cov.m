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
    z_vel_cur = z_cur(2:end,:) - z_cur(1:end-1,:);
    z_data = [z_data; z_cur];
    z_vel = [z_vel; z_vel_cur];
end

z_mean = mean(z_data);
z_cov = cov(z_data);

z_vel_mean = mean(z_vel);
z_vel_cov = cov(z_vel);

save(fullfile(result_path, 'z_info'), 'z_data', 'z_mean', 'z_cov', ...
    'z_vel', 'z_vel_mean', 'z_vel_cov');
disp('z_info is saved')

% bar(1:size_z, z_mean);
figure(1); clf;
boxplot(z_data);
grid on;
title(sprintf('Distribution of Z: %s', dataset));
xlabel('latent axis');
ylabel('value at each axis');

figure(2); clf;
imagesc(abs(z_cov));
colormap('gray');
title(sprintf('Absolute values of elements in Cov. matrix: %s', dataset));

figure(3); clf;
imagesc(z_cov);
colormap('jet');
title(sprintf('Cov. matrix: %s', dataset));

% velocity display
figure(5); clf;
boxplot(z_vel);
grid on;
title(sprintf('Distribution of velocities: %s', dataset));
xlabel('latent axis');
ylabel('velocity');

figure(6); clf;
imagesc(abs(z_vel_cov));
colormap('gray');
title(sprintf('Absolute values of elements in Cov. matrix of velocity: %s', dataset));

figure(7); clf;
imagesc(z_vel_cov);
colormap('jet');
title(sprintf('Cov. matrix of velocity: %s', dataset));


%()()
%('')HAANJU.YOO