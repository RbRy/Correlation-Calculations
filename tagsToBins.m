function [ tags_out ] = tagsToBins( tags_in, bin_width )
%TAGSTOBINS Summary of this function goes here
%   Detailed explanation goes here
    tags_out = cell(length(tags_in),1);
    for i = 1:length(tags_in)
        %tags_out{i} = int64(ceil(double(tags_in{i})*82.3e-12/bin_width));
        tags_out{i} = tagsToBins_mex(tags_in{i},bin_width);
    end

end

