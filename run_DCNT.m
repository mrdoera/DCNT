function results = run_DCNT(seq,res_path, bSaveImage)

varargin=cell(1,2);

varargin(1,1)={'train'};
varargin(1,2)={struct('gpus', [])};

% run ../../matlab/vl_setupnn ;

opts.expDir = 'exp/' ;
opts.dataDir = 'F:\caffe-master\Data\' ;
opts.modelType = 'tracking' ;
opts.sourceModelPath = 'F:\caffe-master\model\';
opts.train = struct() ;
% opts.beta1 = 0.9;
% opts.beta2 = 0.999;
% opts.eps = 1e-8;
[opts, varargin] = vl_argparse(opts, varargin) ;

% experiment setup
% opts.imdbPath = fullfile(opts.expDir, 'imdb.mat') ;
% opts.imdbStatsPath = fullfile(opts.expDir, 'imdbStats.mat') ;
opts.vocEdition = '11' ;
opts.vocAdditionalSegmentations = false ;

global resize;
display = 0;

% g=gpuDevice(1);
% clear g;                             

% test_seq = 'Singer2';
[config] = config_list(seq);

% Image file names
img_files = seq.s_frames;
% Seletected target size
target_sz = [seq.init_rect(1,4), seq.init_rect(1,3)];
% Initial target position
pos       = [seq.init_rect(1,2), seq.init_rect(1,1)] + floor(target_sz/2);

% ================================================================================
% Main entry function for visual tracking
% ================================================================================
% [positions, time] = D_tracking(video_path, img_files, pos, target_sz, ...
%     padding, lambda, output_sigma_factor, interp_factor, ...
%     cell_size, show_visualization);

result = D_tracking(opts,config,display,varargin); 
positions = result;

% ================================================================================
% Return results to benchmark, in a workspace variable
% ================================================================================
rects      = [positions(:,2) - target_sz(2)/2, positions(:,1) - target_sz(1)/2];
rects(:,3) = target_sz(2);
rects(:,4) = target_sz(1);
results.type   = 'rect';
results.res    = rects;
results.fps    = numel(img_files)/time;

end

