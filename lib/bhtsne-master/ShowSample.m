function ShowSample(data_index, dataset, path, tracking_table)

fig = figure(data_index); clf;

test_sample_stride = 5;
train_sample_stride = 2;

if tracking_table(data_index,3) == 1
    video_type = 'testing_videos';
    video_name = sprintf('%02d',tracking_table(data_index,1));
    idx_first_image = test_sample_stride*(tracking_table(data_index,2)+1);
    idx_middle_image = idx_first_image + 5; 
    idx_last_image = idx_first_image + 10;
    
    sec = tracking_table(data_index,2) * test_sample_stride / 25;
    min = round(sec / 60);
    sec = round(rem(sec, 60));
    fprintf('test video %d, %d min %02d sec\n', ...
        tracking_table(data_index,1), min, sec);
    
elseif tracking_table(data_index,3) == 0
    video_type = 'training_videos';
    video_name = sprintf('%02d',tracking_table(data_index,1));
    idx_first_image = train_sample_stride*(tracking_table(data_index,2)+1);
    idx_middle_image = idx_first_image + 5; 
    idx_last_image = idx_first_image + 10;
    
    sec = tracking_table(data_index,2) * train_sample_stride / 25;
    min = round(sec / 60);
    sec = round(rem(sec, 60));
    fprintf('train video %d, %d min %02d sec\n', ...
        tracking_table(data_index,2), min, sec);
    
end
image_path = fullfile(path,dataset,video_type,video_name);

image_file_name = sprintf('frame_%05d.png',idx_first_image);
[X1, image1] = imread(fullfile(image_path, image_file_name));

image_file_name = sprintf('frame_%05d.png',idx_middle_image);
[X2, image2] = imread(fullfile(image_path, image_file_name));

image_file_name = sprintf('frame_%05d.png',idx_last_image);
[X3, image3] = imread(fullfile(image_path, image_file_name));

subplottight(1,3,1), imshow(X1,image1, 'border', 'tight');
subplottight(1,3,2), imshow(X2,image2, 'border', 'tight');
subplottight(1,3,3), imshow(X3,image3, 'border', 'tight');

set(fig, 'Position', [500, 600, 3*size(X1, 2), size(X1, 1)]);

end