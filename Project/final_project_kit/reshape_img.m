%% CIS 520 Final Project

% Description: Reshape 1x30000 image into 100x100x3 image
% Input: 1x30000 
% Output: 100x100x3 uint8 matrix

function output_img = reshape_img(input_img)
    output_img = zeros(100, 100, 3);
    output_img(:,:,1) = reshape(input_img(1:3:end), [100, 100])';
    output_img(:,:,2) = reshape(input_img(2:3:end), [100, 100])';
    output_img(:,:,3) = reshape(input_img(3:3:end), [100, 100])';
    output_img = uint8(output_img);
end