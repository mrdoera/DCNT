function runDCNT_vot

	cleanup = onCleanup(@() exit() ); % Tell Matlab to exit once the function exits
    RandStream.setGlobalStream(RandStream('mt19937ar', 'Seed', sum(clock))); % Set random seed to a different value every time as required by the VOT rules

    [handle, image_init, region_init] = vot('rectangle'); % Obtain communication object
    [cx, cy, w, h] = get_axis_aligned_BB(region_init);
    region = [cx, cy, w, h];
	
	varargin=cell(1,2);
	varargin(1,1)={'train'};
	varargin(1,2)={struct('gpus', [])};

	% run ../../matlab/vl_setupnn ;

	opts.expDir = 'exp/' ;
	opts.dataDir = 'F:\caffe-master\Data\' ;
	opts.modelType = 'tracking' ;
	opts.sourceModelPath = 'F:\caffe-master\model\';
	opts.train = struct() ;
	[opts, varargin] = vl_argparse(opts, varargin) ;

	% experiment setup
	opts.vocEdition = '11' ;
	opts.vocAdditionalSegmentations = false ;

	global resize = 100;
	display = 0;

    config.imgList=imgList;
    config.gt=gt;
    config.nFrames=nFrames;
    config.name=test_seq;
	
	% Seletected target size
	% target_sz = round([region(4) region(3)]);
	% Initial target position
	% pos = [region(2) region(1)];

	global objSize;

	LocGt=config.gt;
	num_channels=64;
	output_sigma_factor = 0.1;

	%--------------------VGG16---------------------------
	[net1,avgImg]=initVGG16Net();
	index=[17 23 29];
	
	nFrame=config.nFrames;
	[Gt]=config.gt;
	
	% Load Image 
	im1 = imread(image_init);% img_files = seq.s_frames;
	scale = 1;
	objSize = round([region(4) region(3)]);
	
	
	frame = 1;
	while true
		% for i=2:nFrame 
		if frame > 1
			% im1=imread(config.imgList{i});
			[handle, image] = handle.frame(handle); % Get the next frame
			if isempty(image)
				break;
			end
			%load image
			im1 = imread(image);
			im=imresize(im1,scale);    
			if size(im1,3)==1
				im = cat(3, im, im, im);
				im1 = cat(3, im1, im1, im1);
			end    
				
			patch = get_subwindow(im, pos, window_sz);       
			patch1 = single(patch) - meanImg;    
			net1.eval({'input',patch1}); % extract features
			
			feat = cell(length(index),1);
			featPCA = cell(length(index),1);
			for j=1:length(index)
				feat1 = gather(net1.vars(index(j)).value);
				feat1 = imResample(feat1, sz_window(1:2));
				feat{j} = bsxfun(@times, feat1, cos_window);    
				[hf,wf,cf]=size(feat{j});
				feat_ = reshape(feat{j},hf*wf,cf);
				feat_ = feat_*coeff{j};                
				featPCA{j} = reshape(feat_,hf,wf,num_channels);        
			end
		%--------------------------  Prediction  ---------------------------------  
			net_online.eval({'input1',featPCA{1},'input2',featPCA{2},'input3',featPCA{3}, ...
				'input4',featPCA1st});   
			regression_map=gather(net_online.vars(22).value);        
					 
			motion_sigma = target_sz1*motion_sigma_factor;    
			motion_map = gaussian_shaped_labels(motion_sigma, l1_patch_num);
				
			response=regression_map.*motion_map;    
			
			[vert_delta, horiz_delta] = find(response == max(response(:)), 1);
			vert_delta  = vert_delta  - ceil(hf/2);
			horiz_delta = horiz_delta - ceil(wf/2);   
					  
			pos = pos + cell_size * [vert_delta, horiz_delta];
					   
			target_szU = scale_estimation(im,pos,target_szU,window_sz,...
					net1,net_online,coeff,meanImg,featPCA1st);            
									
			targetLoc = [pos([2,1]) - target_szU([2,1])/2, target_szU([2,1])];    
			% result(i,:) = round(targetLoc/scale);    
			region = round(targetLoc/scale);
			handle = handle.report(handle, region);
			
			label1 = imresize(regression_map,[size(im1,1) size(im1,1)])*255;
			patch1 = imresize(patch,[size(im1,1) size(im1,1)]);     
			imd = [im1];
			%-----------Display current frame-----------------
			if display   
				hc = get(gca, 'Children'); delete(hc(1:end-1));
				set(hd,'cdata',imd); hold on;                                
				rectangle('Position', result(i,:), 'EdgeColor', [0 0 1], 'Linewidth', 1);                       
				set(gca,'position',[0 0 1 1]);
				text(10,10,num2str(i),'Color','y', 'HorizontalAlignment', 'left', 'FontWeight','bold', 'FontSize', 30); 
				hold off;
				drawnow;  
			end
			
			%-----------Model update-------------------------                                
			labelU = circshift(label,[vert_delta,horiz_delta]);
			feat_update{cur} = featPCA;              
			label_update{cur} = labelU; 
			if cur==num_update    
				trainOpts.batchSize = 1 ;
				trainOpts.numSubBatches = 1 ;
				trainOpts.continue = true ;
				trainOpts.gpus = [] ;
				trainOpts.prefetch = true ;

				trainOpts.expDir = 'exp/update/' ;
				trainOpts.learningRate = 2e-9;        
				trainOpts.weightDecay= 1;
				trainOpts.numEpochs = 2;

				train=1;
				imdb=[];
				input={feat_update featPCA1st label_update};
				opts.train.gpus=[];
				bopts.useGpu = numel(opts.train.gpus) > 0 ;
									
				info = cnn_train_dag_update(net_online, imdb, input,getBatchWrapper(bopts), ...
									 trainOpts, ...
									 'train', train, ...                     
									 opts.train) ;      
				cur=1;        
			else 
				cur=cur+1;            
			end					   
		end	
		
		if objSize(1)*objSize(2)>resize*resize
			scale = resize/max(objSize);    
			disp('resized');
		end
		im = imresize(im1,scale);
		cell_size = 4;
		if size(im,3) == 1
			im = cat(3, im, im, im);
			im1 = cat(3, im1, im1, im1);
		end
		targetLoc = region;
		target_sz = [targetLoc(4) targetLoc(3)];
		im_sz = size(im);
		window_sz = get_search_window(target_sz,im_sz);
		l1_patch_num = ceil(window_sz/cell_size);
		l1_patch_num = l1_patch_num-mod(l1_patch_num,2)+1;
		cos_window = hann(l1_patch_num(1))*hann(l1_patch_num(2))';
		sz_window = size(cos_window);
		
		pos = [targetLoc(2), targetLoc(1)];
		patch = get_subwindow(im, pos, window_sz);%get_subwindow
		meanImg=zeros(size(patch));
		meanImg(:,:,1)=avgImg(1);
		meanImg(:,:,2)=avgImg(2);
		meanImg(:,:,3)=avgImg(3);
		patch1 = single(patch) - meanImg;% patches searched 
		%-----------extract features-------------------
		net1.eval({'input',patch1});
		% index=[17 23 29];
		feat = cell(length(index),1);
		featPCA = cell(length(index),1);
		coeff = cell(length(index),1);
		for i=1:length(index)
			feat1 = gather(net1.vars(index(i)).value);
			feat1 = imResample(feat1, sz_window(1:2));
			feat{i} = bsxfun(@times, feat1, cos_window);
			
			[hf,wf,cf]=size(feat{i});
			matrix = reshape(feat{i},hf*wf,cf);
			coeff_m = pca(matrix);
			coeff{i} = coeff_m(:,1:num_channels);

			feat_ = reshape(feat{i},hf*wf,cf);
			feat_ = feat_*coeff{i};                
			featPCA{i} = reshape(feat_,hf,wf,num_channels);
		end
		target_sz1=ceil(target_sz/cell_size);
		output_sigma = target_sz1*output_sigma_factor;
		label=gaussian_shaped_labels(output_sigma, l1_patch_num);

		label1=imresize(label,[size(im1,1) size(im1,1)])*255;
		patch1=imresize(patch,[size(im1,1) size(im1,1)]);
		imd=[im1];

		%-------------------Display First frame----------
		if display    
			figure(2);
			set(gcf,'Position',[200 300 480 320],'MenuBar','none','ToolBar','none');
			hd = imshow(imd,'initialmagnification','fit'); hold on;
			rectangle('Position', Gt(1,:), 'EdgeColor', [0 0 1], 'Linewidth', 1);    
			set(gca,'position',[0 0 1 1]);
			text(10,10,'1','Color','y', 'HorizontalAlignment', 'left', 'FontWeight','bold', 'FontSize', 30);
			hold off;
			drawnow;   
		end
		if frame == 1
			%-------------------first frame initialization-----------
			trainOpts.numEpochs=4000;
			net_online=initNet(target_sz1);%Residual learning
			% net_online.mode = 'test';

			trainOpts.batchSize = 1 ;
			trainOpts.numSubBatches = 1 ;
			trainOpts.continue = true ;
			trainOpts.gpus = [] ;
			trainOpts.prefetch = true ;

			trainOpts.expDir = opts.expDir ;
			trainOpts.learningRate=5e-8;
			trainOpts.weightDecay= 1;

			train=1;
			imdb=[];
			featPCA1st = featPCA{3};                      %  to be determined
			input={featPCA{1} featPCA{2} featPCA{3} featPCA1st label};

			opts.train.gpus=[];
			bopts.useGpu = numel(opts.train.gpus) > 0 ;

			net_online.conserveMemory=0;
			net_online.eval({'input1',input{1},'input2',input{2},'input3',input{3},'input4',featPCA1st});
			info = cnn_train_dag(net_online, imdb, input,getBatchWrapper(bopts), ...
								 trainOpts, ...
								 'train', train, ...                     
								 opts.train);

			%----------------online prediction------------------
			motion_sigma_factor=0.6;
			cell_size=4;
			global num_update;
			num_update=2;
			cur=1;
			feat_update=cell(num_update,1);
			label_update=cell(num_update,1);
			target_szU = target_sz;
		end
		frame = frame +1;
    end
	handle.quit(handle);
end

function [cx, cy, w, h] = get_axis_aligned_BB(region)

% GETAXISALIGNEDBB extracts an axis aligned bbox from the ground truth REGION with same area as the rotated one
	nv = numel(region);
	assert(nv==8 || nv==4);

	if nv==8
		cx = mean(region(1:2:end));
		cy = mean(region(2:2:end));
		x1 = min(region(1:2:end));
		x2 = max(region(1:2:end));
		y1 = min(region(2:2:end));
		y2 = max(region(2:2:end));
		A1 = norm(region(1:2) - region(3:4)) * norm(region(3:4) - region(5:6));
		A2 = (x2 - x1) * (y2 - y1);
		s = sqrt(A1/A2);
		w = s * (x2 - x1) + 1;
		h = s * (y2 - y1) + 1;
	else
		x = region(1);
		y = region(2);
		w = region(3);
		h = region(4);
		cx = x+w/2;
		cy = y+h/2;
	end

end

function fn = getBatchWrapper(opts)
% -------------------------------------------------------------------------
fn = @(imdb,batch) getBatch(imdb,batch,false,opts,'prefetch',nargout==0) ;
end