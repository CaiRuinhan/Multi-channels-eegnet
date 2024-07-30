folderPath = ('D:\brain data');
fileList = dir(fullfile(folderPath,'*.easy'));
eye_artifact_threshold = 0.8;
muscle_artifact_threshold = 0.8;
linenoise_artifact_threshold = 0.8;
channelnoise_artifact_threshold = 0.8;

for i = 1:length(fileList)
    filePath = fullfile(folderPath,fileList(i).name);
    EEG = pop_easy(filePath,0,1,'');
    EEG = pop_chanedit(EEG,'lookup','D:\tool\matlab\toolbox\eeglab\plugins\dipfit\standard_BEM\elec\standard_1005.elc');
    EEG = pop_select(EEG,'rmchannel',{'X','Y','Z'});
    EEG = pop_eegfiltnew(EEG,'locutoff',0.1,'hicutoff',46);
    % EEG = pop_eegfiltnew(EEG,'locutoff',47.5,'hicutoff',52,'revfilt',1);
    EEG = pop_rmbase(EEG, []);
    EEG = pop_runica(EEG,'extended',1,'interrupt','on');
    EEG = eeg_checkset(EEG);
    % 使用iclabel对ica成分进行分类
    EEG = pop_iclabel(EEG,'default');
    % 获取每个成分的分类概率
    iclabel_probabilities = EEG.etc.ic_classification.ICLabel.classifications;
% ICLabel 的分类结果是一个矩阵，其中每一对应一个ICA成分，每一列对应一个分类类别的概率。
% 具体的类别顺序如下：
% 1 脑电（Brain）
% 2 肌电（Muscle）
% 3 眼电（Eye）
% 4 心电（Heart）
% 5 线噪声（Line Noise）
% 6 伪迹（Channel Noise）
% 7 其他（Other）
    % 检测眼电
    eye_artifact_comps = find(iclabel_probabilities(:, 3) > eye_artifact_threshold);
    muscle_artifact_comps = find(iclabel_probabilities(:, 2) > muscle_artifact_threshold);
    linenoise_artifact_comps = find(iclabel_probabilities(:, 5) > linenoise_artifact_threshold);
    channelnoise_artifact_comps = find(iclabel_probabilities(:, 6) > channelnoise_artifact_threshold);
    all_artifact_comps = union(eye_artifact_comps, muscle_artifact_comps,linenoise_artifact_comps,channelnoise_artifact_comps);
    EEG = pop_subcomp(EEG,all_artifact_comps,0);
    EEG = eeg_checkset(EEG);
    index = strfind(fileName,'_');
    newFileName = strcat(fileName(index+1:end-10),'.set');
    EEG = pop_saveset(EEG,'filename',newFileName, 'filepath','D:/预处理');
end

% 
% eye_artifact_threshold = 0.8;
% muscle_artifact_threshold = 0.8;
% fileName = 'E:\脑电数据\brain data\20240514145830_DM-1_test.easy';
% EEG = pop_easy(fileName,0,1,'');
% EEG = pop_chanedit(EEG,'lookup','E:\MATLAB\toolbox\eeglab2023.0\plugins\dipfit\standard_BEM\elec\standard_1005.elc');
% EEG = pop_select(EEG,'rmchannel',{'X','Y','Z'});
% EEG = pop_eegfiltnew(EEG,'locutoff',1,'hicutoff',46);
% % EEG = pop_eegfiltnew(EEG,'locutoff',48,'hicutoff',52,'revfilt',1);
% EEG = pop_rmbase(EEG, []);
% EEG = eeg_checkset(EEG);
% EEG = pop_rmbase(EEG, []);
% EEG = pop_runica(EEG,'extended',1,'interrupt','on');
% EEG = eeg_checkset(EEG);
% % 使用iclabel对ica成分进行分类
% EEG = pop_iclabel(EEG,'default');
% % 获取每个成分的分类概率
% iclabel_probabilities = EEG.etc.ic_classification.ICLabel.classifications;
% eye_artifact_comps = find(iclabel_probabilities(:, 3) > eye_artifact_threshold);
% muscle_artifact_comps = find(iclabel_probabilities(:, 2) > muscle_artifact_threshold);
% all_artifact_comps = union(eye_artifact_comps, muscle_artifact_comps);
% if ~isfield(EEG, 'reject')
%     EEG.reject = struct();
% end
% 
% % 确保 EEG.reject 结构体包含 gcompreject 字段
% if ~isfield(EEG.reject, 'gcompreject')
%     EEG.reject.gcompreject = [];
% end
% EEG = pop_subcomp(EEG,all_artifact_comps,0);
% disp(iclabel_probabilities);
% index = strfind(fileName,'_');
% newFileName = strcat(fileName(index+1:end-10),'.set');
% EEG = pop_saveset(EEG,'filename',newFileName, 'filepath','E:/脑电数据/预处理');
