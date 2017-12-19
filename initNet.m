function net_online = initNet(target_sz1)
%Init network
channel=64;

rw=ceil(target_sz1(2)/2);
rh=ceil(target_sz1(1)/2);
fw=2*rw+1;
fh=2*rh+1;

net_online=dagnn.DagNN();
%----------------------- Base mapping -----------------------------------%
net_online.addLayer('conv11', dagnn.Conv('size', [fw,fh,channel,1],...
    'hasBias', true, 'pad',...
    [rh,rh,rw,rw], 'stride', [1,1]), 'input1', 'conv_11', {'conv11_f', 'conv11_b'});

f = net_online.getParamIndex('conv11_f') ;
net_online.params(f).value=single(randn(fh,fw,channel,1) /...
    sqrt(rh*rw*channel))/1e8;
net_online.params(f).learningRate=1;
net_online.params(f).weightDecay=1e3;

f = net_online.getParamIndex('conv11_b') ;
net_online.params(f).value=single(zeros(1,1));
net_online.params(f).learningRate=2;
net_online.params(f).weightDecay=1e3;
% % % % % % % % % % % % % % % % % % % % % % % % % % % 
% ADDLAYER(NAME, LAYER, INPUTS, OUTPUTS, PARAMS) adds the specified layer to the network. 
% NAME is a string with the layer name, used as a unique indentifier. 
% BLOCK is the object implementing the layer, which should be a subclass of the Layer. 
% INPUTS, OUTPUTS are cell arrays of variable names, 
% and PARAMS of parameter names.
% % % % % % % % % % % % % % % % % % % % % % % % % % % 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%----------------------- Residual mapping  -------------------------------%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%-----------------------       Res1  f5_3       -------------------------------%
net_online.addLayer('conv21', dagnn.Conv('size', [1,1,channel,channel],...
    'hasBias', true, 'pad',...
    [0,0,0,0], 'stride', [1,1]), 'input1', 'conv_21', {'conv21_f', 'conv21_b'});
net_online.addLayer('relu21', dagnn.ReLU(), 'conv_21', 'relu_21');

f = net_online.getParamIndex('conv21_f') ;
net_online.params(f).value=single(randn(1,1,channel,channel) /...
    sqrt(1*1*channel))/1e8;
net_online.params(f).learningRate=1;
net_online.params(f).weightDecay=1;

f = net_online.getParamIndex('conv21_b') ;
net_online.params(f).value=single(zeros(channel,1));
net_online.params(f).learningRate=2;
net_online.params(f).weightDecay=1;

net_online.addLayer('conv22', dagnn.Conv('size', [1,1,channel,channel],...
    'hasBias', true, 'pad',...
    [0,0,0,0], 'stride', [1,1]), 'relu_21', 'conv_22', {'conv22_f', 'conv22_b'});
net_online.addLayer('relu22', dagnn.ReLU(), 'conv_22', 'relu_22');

f = net_online.getParamIndex('conv22_f') ;
net_online.params(f).value=single(randn(1,1,channel,channel) /...
    sqrt(1*1*channel));
net_online.params(f).learningRate=1;
net_online.params(f).weightDecay=1;

f = net_online.getParamIndex('conv22_b') ;
net_online.params(f).value=single(zeros(channel,1));
net_online.params(f).learningRate=2;
net_online.params(f).weightDecay=1;

net_online.addLayer('conv23', dagnn.Conv('size', [1,1,channel,1],...
    'hasBias', true, 'pad',...
    [0,0,0,0], 'stride', [1,1]), 'relu_22', 'conv_23', {'conv23_f', 'conv23_b'});

f = net_online.getParamIndex('conv23_f') ;
net_online.params(f).value=single(randn(1,1,channel,1) /...
    sqrt(1*1*channel));
net_online.params(f).learningRate=1;
net_online.params(f).weightDecay=1;

f = net_online.getParamIndex('conv23_b') ;
net_online.params(f).value=single(zeros(1,1));
net_online.params(f).learningRate=2;
net_online.params(f).weightDecay=1;
%-----------------------       Res2  f4_3      -------------------------------%
net_online.addLayer('conv31', dagnn.Conv('size', [1,1,channel,channel],...
    'hasBias', true, 'pad',...
    [0,0,0,0], 'stride', [1,1]), 'input2', 'conv_31', {'conv31_f', 'conv31_b'});
net_online.addLayer('relu31', dagnn.ReLU(), 'conv_31', 'relu_31');

f = net_online.getParamIndex('conv31_f') ;
net_online.params(f).value=single(randn(1,1,channel,channel) /...
    sqrt(1*1*channel))/1e8;
net_online.params(f).learningRate=1;
net_online.params(f).weightDecay=1;

f = net_online.getParamIndex('conv31_b') ;
net_online.params(f).value=single(zeros(channel,1));
net_online.params(f).learningRate=2;
net_online.params(f).weightDecay=1;

net_online.addLayer('conv32', dagnn.Conv('size', [1,1,channel,channel],...
    'hasBias', true, 'pad',...
    [0,0,0,0], 'stride', [1,1]), 'relu_31', 'conv_32', {'conv32_f', 'conv32_b'});
net_online.addLayer('relu32', dagnn.ReLU(), 'conv_32', 'relu_32');

f = net_online.getParamIndex('conv32_f') ;
net_online.params(f).value=single(randn(1,1,channel,channel) /...
    sqrt(1*1*channel));
net_online.params(f).learningRate=1;
net_online.params(f).weightDecay=1;

f = net_online.getParamIndex('conv32_b') ;
net_online.params(f).value=single(zeros(channel,1));
net_online.params(f).learningRate=2;
net_online.params(f).weightDecay=1;

net_online.addLayer('conv33', dagnn.Conv('size', [1,1,channel,1],...
    'hasBias', true, 'pad',...
    [0,0,0,0], 'stride', [1,1]), 'relu_32', 'conv_33', {'conv33_f', 'conv33_b'});

f = net_online.getParamIndex('conv33_f') ;
net_online.params(f).value=single(randn(1,1,channel,1) /...
    sqrt(1*1*channel));
net_online.params(f).learningRate=1;
net_online.params(f).weightDecay=1;

f = net_online.getParamIndex('conv33_b') ;
net_online.params(f).value=single(zeros(1,1));
net_online.params(f).learningRate=2;
net_online.params(f).weightDecay=1;
%-----------------------       Res3 f3_3        -------------------------------%
% net_online.addLayer('pool41', dagnn.Pooling('poolSize', [2,2], 'pad',...
%     [0,1,0,1], 'stride', [2,2]),'input3','pool_41');       % down-sampling  max pooling

net_online.addLayer('conv41', dagnn.Conv('size', [1,1,channel,channel],...
    'hasBias', true, 'pad',...
    [0,0,0,0], 'stride', [1,1]), 'input3', 'conv_41', {'conv41_f', 'conv41_b'});
net_online.addLayer('relu41', dagnn.ReLU(), 'conv_41', 'relu_41');

f = net_online.getParamIndex('conv41_f') ;
net_online.params(f).value=single(randn(1,1,channel,channel) /...
    sqrt(1*1*channel))/1e8;
net_online.params(f).learningRate=1;
net_online.params(f).weightDecay=1;

f = net_online.getParamIndex('conv41_b') ;
net_online.params(f).value=single(zeros(channel,1));
net_online.params(f).learningRate=2;
net_online.params(f).weightDecay=1;

net_online.addLayer('conv42', dagnn.Conv('size', [1,1,channel,channel],...
    'hasBias', true, 'pad',...
    [0,0,0,0], 'stride', [1,1]), 'relu_41', 'conv_42', {'conv42_f', 'conv42_b'});
net_online.addLayer('relu42', dagnn.ReLU(), 'conv_42', 'relu_42');

f = net_online.getParamIndex('conv42_f') ;
net_online.params(f).value=single(randn(1,1,channel,channel) /...
    sqrt(1*1*channel));
net_online.params(f).learningRate=1;
net_online.params(f).weightDecay=1;

f = net_online.getParamIndex('conv42_b') ;
net_online.params(f).value=single(zeros(channel,1));
net_online.params(f).learningRate=2;
net_online.params(f).weightDecay=1;

net_online.addLayer('conv43', dagnn.Conv('size', [1,1,channel,1],...
    'hasBias', true, 'pad',...
    [0,0,0,0], 'stride', [1,1]), 'relu_42', 'conv_43', {'conv43_f', 'conv43_b'});

f = net_online.getParamIndex('conv43_f') ;
net_online.params(f).value=single(randn(1,1,channel,1) /...
    sqrt(1*1*channel));
net_online.params(f).learningRate=1;
net_online.params(f).weightDecay=1;

f = net_online.getParamIndex('conv43_b') ;
net_online.params(f).value=single(zeros(1,1));
net_online.params(f).learningRate=2;
net_online.params(f).weightDecay=1;

%----------------------- Temporal residual -----------------------------------%
net_online.addLayer('conv51', dagnn.Conv('size', [1,1,channel,1],...
    'hasBias', true, 'pad',...
    [0,0,0,0], 'stride', [1,1]), 'input4', 'conv_51', {'conv51_f', 'conv51_b'});

f = net_online.getParamIndex('conv51_f') ;
net_online.params(f).value=single(randn(1,1,channel,1) /...
    sqrt(1*1*channel))/1e10;
net_online.params(f).learningRate=1e-2;
net_online.params(f).weightDecay=1e3;

f = net_online.getParamIndex('conv51_b') ;
net_online.params(f).value=single(zeros(1,1));
net_online.params(f).learningRate=2e-2;
net_online.params(f).weightDecay=1e3;


net_online.addLayer('sum1',dagnn.Sum(),{'conv_11','conv_23','conv_33','conv_43','conv_51'},'sum_1');

net_online.addLayer('L2Loss',...
    RegressionL2Loss(),{'sum_1','label'},'objective');

end
