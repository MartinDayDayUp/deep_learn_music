


fc = 300e6;
lambda = physconst("LightSpeed")/fc;
ula = phased.ULA(NumElements=8,ElementSpacing=lambda/2);
nsanpshots = 200;
angleGrid = -90:1:90;


rng("default")
[signalsTrain,scovsTrain,anglesTrain] = helperGenerateTrainData(NoisePower=[1,2,5,7], ...
    Shuffle=true, ...
    ULA=ula, ...
    Nsnapshots=nsanpshots);
ndata = length(anglesTrain);
sigCov = zeros([size(signalsTrain,2) size(signalsTrain,2) 1 size(signalsTrain,3)]);
sigCov(:,:,1,:) = real(scovsTrain);
sigCov(:,:,2,:) = imag(scovsTrain);

anglesBinaryVec = zeros([length(angleGrid) size(signalsTrain,3) ]);
for ii = 1:ndata
    angleSet = anglesTrain{ii};
    anglesBinaryVec(:,ii) = ismember(angleGrid,angleSet);
end


% 调用函数保存数据
saveDatasetToCsv(scovsTrain, anglesBinaryVec, 'dataCsv.csv', 'labelCsv.csv');




function saveDatasetToCsv(scovsTrain, anglesBinaryVec, dataFileName, labelFileName)
    % 获取样本总数
    N = size(scovsTrain, 3); % 65884
    
    % --- 处理 scovsTrain ---
    % 提取实部和虚部
    realPart = real(scovsTrain); % 8x8x65884
    imagPart = imag(scovsTrain); % 8x8x65884
    
    % 将 8x8 的矩阵展平为 64 维向量。
    % 注意：MATLAB 是列优先（按列展开），PyTorch 是行优先。
    % 在 PyTorch 中读取后，如果需要还原为 8x8，需要调整 reshape 逻辑。
    % 这里按照 MATLAB 默认的 (:) 方式展开为 64 维。
    realFeatures = reshape(realPart, [], N)'; % N x 64
    imagFeatures = reshape(imagPart, [], N)'; % N x 64
    
    % 拼接实部和虚部：先实后虚 -> N x 128
    dataMatrix = [realFeatures, imagFeatures];
    
    % --- 处理 anglesBinaryVec ---
    % anglesBinaryVec 为 181x65884，需要转置为 N x 181
    labelMatrix = anglesBinaryVec'; 
    
    % --- 写入 CSV ---
    % 使用 writematrix 写入 CSV，通常不写表头以方便 PyTorch 读取
    writematrix(dataMatrix, dataFileName);
    writematrix(labelMatrix, labelFileName);
    
    fprintf('数据保存成功：\n');
    fprintf('  %s: %d x %d\n', dataFileName, size(dataMatrix, 1), size(dataMatrix, 2));
    fprintf('  %s: %d x %d\n', labelFileName, size(labelMatrix, 1), size(labelMatrix, 2));
end





function [Signals,Scovs, Angles, ula, angleRes] = helperGenerateTrainData(Nvargs)
% This function is only intended to support this example. It may be changed
% or removed in a future release. 
arguments
    Nvargs.NoisePower = 1
    Nvargs.ULA = []
    Nvargs.Fc = 300e6
    Nvargs.NAngles = 181
    Nvargs.NSources = [1,2]
    Nvargs.Shuffle logical = true 
    Nvargs.RandomSample = false
    Nvargs.Nsnapshots = 200
    Nvargs.Ndata = 1;
end

Nsnapshots = Nvargs.Nsnapshots; % Number of snapshots
fc = 300e6;
ula = Nvargs.ULA;
Nelements = ula.NumElements;
lambda = physconst("LightSpeed")/fc;
pos = getElementPosition(ula) / lambda; % Element positions in wavelengths

noisePwrs = Nvargs.NoisePower;

% Angle resolution and combinations
nangle = Nvargs.NAngles;
angleRes = linspace(-90, 90, nangle);
angleCombsList = cell(length(Nvargs.NSources));
for ii = 1:length(Nvargs.NSources)
    nsources = Nvargs.NSources(ii);
    angleCombs = nchoosek(angleRes, nsources); % All C(2, 181) combinations
    angleCombsList{ii} = mat2cell(angleCombs,ones(1,length(angleCombs)),nsources);
end
angleCombsAll =  cat(1,angleCombsList{:});

if Nvargs.Shuffle
    angleCombsAll = angleCombsAll(randperm(size(angleCombsAll, 1)), :);
    
end

nsnr = length(noisePwrs);
Signals = zeros(Nsnapshots, Nelements, size(angleCombsAll, 1)*nsnr);
Scovs = zeros(Nelements, Nelements, size(angleCombsAll, 1)*nsnr);
Angles = cell(size(angleCombsAll, 1)*nsnr, 1);

% Generate signals for each angle pair
for ii = 1:size(angleCombsAll, 1)
    az_ang = angleCombsAll{ii, :}; % Select the angle pair
    el_ang = zeros(1, numel(az_ang)); % Elevation is zero
    for jj = 1:nsnr
        noisePwr = noisePwrs(jj);
        % Generate signal using the sensorsig function
        [signal,~,scov] = sensorsig(pos, Nsnapshots, [az_ang; el_ang], noisePwr);
        idx = (ii-1)*nsnr+jj;
        Signals(:, :, idx) = signal;
        Scovs(:, :, idx) = scov;
        Angles{idx} = az_ang;
    end
end
end
