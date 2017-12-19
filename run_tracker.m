function [precision, fps] = run_tracker(video, show_visualization, show_plots)

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

%     test_seq = 'Singer2';
%     [config] = config_list(test_seq);
% 
%     result = D_tracking(opts,config,display,varargin); 

	%path to the videos (you'll be able to choose one with the GUI).
	base_path = 'F:\caffe-master\Data\';
	%default settings
	if nargin < 1, video = 'choose'; end
	if nargin < 2, show_visualization = ~strcmp(video, 'all'); end
	if nargin < 3, show_plots = ~strcmp(video, 'all'); end

	
    switch video
	case 'choose',
		%ask the user for the video, then call self with that video name.
		video = choose_video(base_path);
		if ~isempty(video),
			[precision, fps] = run_tracker(video,show_visualization, show_plots);
            
			if nargout == 0,  %don't output precision as an argument
				clear precision
			end
		end
				
	case 'all',
		%all videos, call self with each video name.
		
		%only keep valid directory names
		dirs = dir(base_path);
		videos = {dirs.name};
		videos(strcmp('.', videos) | strcmp('..', videos) | ...
			strcmp('anno', videos) | ~[dirs.isdir]) = [];
		
		%the 'Jogging' sequence has 2 targets, create one entry for each.
		%we could make this more general if multiple targets per video
		%becomes a common occurence.
		videos(strcmpi('Jogging', videos)) = [];
		videos(end+1:end+2) = {'Jogging.1', 'Jogging.2'};
		
		all_precisions = zeros(numel(videos),1);  %to compute averages
		all_fps = zeros(numel(videos),1);
		
		if ~exist('matlabpool', 'file'),
			%no parallel toolbox, use a simple 'for' to iterate
			for k = 1:numel(videos),
				[all_precisions(k), all_fps(k)] = run_tracker(videos{k}, show_visualization, show_plots);
			end
		else
			%evaluate trackers for all videos in parallel
			if matlabpool('size') == 0,
				matlabpool open;
			end
			parfor k = 1:numel(videos),
				[all_precisions(k), all_fps(k)] = run_tracker(videos{k}, show_visualization, show_plots);
			end
		end
		
		%compute average precision at 20px, and FPS
		mean_precision = mean(all_precisions);
		fps = mean(all_fps);
		fprintf('\nAverage precision (20px):% 1.3f, Average FPS:% 4.2f\n\n', mean_precision, fps)
		if nargout > 0,
			precision = mean_precision;
        end
		
		
	otherwise
		%we were given the name of a single video to process.
	
		%get image file names, initial state, and ground truth for evaluation
		[img_files, pos, target_sz, ground_truth, video_path] = load_video_info(base_path, video);
		
        
		%call tracker function with all the relevant parameters
		[positions, time] = tracker_lct(video_path, img_files, pos, target_sz, config, show_visualization);
		
		
		%calculate and show precision plot, as well as frames-per-second
		precisions = precision_plot(positions, ground_truth, video, show_plots);
		fps = numel(img_files) / time;

		fprintf('%12s - Precision (20px):% 1.3f, FPS:% 4.2f\n', video, precisions(20), fps)

		if nargout > 0,
			%return precisions at a 20 pixels threshold
			precision = precisions(20);
        end
        
	end
end
