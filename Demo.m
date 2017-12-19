function info = Demo()

varargin=cell(1,2);

varargin(1,1)={'train'};
varargin(1,2)={struct('gpus', [])};

run ../../matlab/vl_setupnn ;
% run F:\caffe-master\CREST-Release-master\matconvnetmatlab/vl_setupnn ;
% addpath ../matconvnet/examples ;

opts.expDir = 'exp/' ;
opts.dataDir = 'F:\caffe-master\Data\' ;%F:\caffe-master\Data   exp/data/
opts.modelType = 'tracking' ;
opts.sourceModelPath = 'F:\caffe-master\model\';%'exp/models/' ;
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
display=1;

% g=gpuDevice(1);
% clear g;                             

test_seq = 'Singer2';
[config] = config_list(test_seq);

result = D_tracking(opts,config,display,varargin);        
       



