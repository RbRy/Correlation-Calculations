function [ tags_out ] = tagsToBins( tags_in, bin_width )
%TAGSTOBINS Summary of this function goes here
%   Detailed explanation goes here
    tags_out = cell(length(tags_in),1);
    for i = 1:length(tags_in)
        tags_out(i) = int64(ceil(double(test{i})*82.3e-12/25e-9));

end

