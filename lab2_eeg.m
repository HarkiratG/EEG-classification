
%% resetting workspace
clc
clearvars
close all

%% extracting data
filedir = '/Users/harki/Google Drive/HOMEWORK/Year 4/4.3/491/Lab 2 - EEG/Data Package/EEG_IMU/';
accel_1 = load(strcat(filedir,'Accelerometer_data_trial1.mat'));
eeg_1 = load(strcat(filedir,'EEG_data_trial1.mat'));
accel_2 = load(strcat(filedir,'Accelerometer_data_trial2.mat'));
eeg_2 = load(strcat(filedir,'EEG_data_trial2.mat'));

accel.Accelerometer = [accel_1.Accelerometer; accel_2.Accelerometer];
accel.time = [accel_1.time; accel_1.time(end) + accel_2.time];
eeg.data = [eeg_1.BioRadioData{:}; eeg_2.BioRadioData{:}]';
eeg.time = [eeg_1.eegtimestamp; eeg_1.eegtimestamp(end) + eeg_2.eegtimestamp]';

%% labeling
datums = accel.time(accel.Accelerometer(:,2) < -0.9);

i = 1:1:size(datums,1)-1;
separation_window = (datums(i+1) - datums(i) > 1);

writing_window = [datums(separation_window) + 2, datums(separation_window) + 4];
eating_window = [datums(separation_window) - 1.5, datums(separation_window) + .5];

eeg.labels = label_data(eeg.time, writing_window, eating_window);
accel.labels = label_data(accel.time, writing_window, eating_window);

%% bandpass filter
bpFilt = designfilt('bandpassfir', 'FilterOrder', 20,...
	'CutoffFrequency1', 1, 'CutoffFrequency2', 45, 'SampleRate', 500);
for i = 1:1:8
	eeg.data_filt(i,:) = filtfilt(bpFilt, eeg.data(i,:));
end

%% extracting useful EEG data
eeg.writing = eeg.data_filt(:,eeg.labels == 1);
eeg.eating  = eeg.data_filt(:,eeg.labels == 2);

%% splitting data into training and testing
eeg.all_x = eeg.data_filt(:,eeg.labels > 0);
eeg.all_y = eeg.data_filt(eeg.labels > 0);

training_size = size(eeg.writing, 2)/size(writing_window,1) * floor(0.75*size(writing_window,1));

eeg.train_writing = eeg.writing(:, 1:training_size);
eeg.train_eating  = eeg.eating (:, 1:training_size);
eeg.train_data    = eeg.all_x  (:, 1:training_size);
eeg.train_labels  = eeg.all_y  (:, 1:training_size);

eeg.test_writing = eeg.writing(:, training_size+1:end);
eeg.test_eating  = eeg.eating (:, training_size+1:end);
eeg.test_data    = eeg.all_x  (:, training_size+1:end);
eeg.test_labels  = eeg.all_y  (:, training_size+1:end);

%% CSP
csp_train = csp(eeg.train_writing, eeg.train_eating);
csp_test  = csp(eeg.test_writing,  eeg.test_eating);

%% LDA
lda.sigma_var_writing = (csp_train.var_writing * csp_train.var_writing')/...
	size(csp_train.var_writing,2);
lda.sigma_var_eating  = (csp_train.var_eating * csp_train.var_eating')/...
	size(csp_train.var_eating,2);

lda.u_writing         = mean(csp_train.var_writing,2);
lda.u_eating          = mean(csp_train.var_eating,2);
lda.u_ave             = mean( [lda.u_writing lda.u_eating], 2 );

lda.sigma_within      = lda.sigma_var_writing + lda.sigma_var_eating;
lda.sigma_between     = ( (lda.u_writing - lda.u_ave) * (lda.u_writing - lda.u_ave)'...
	+ (lda.u_eating - lda.u_ave)*(lda.u_eating - lda.u_ave)') /2;

[lda.eigenvectors, lda.eigenvalues] = eig(lda.sigma_between, lda.sigma_within);

lda.w = lda.eigenvectors(:, end);

lda.projected_writing_train = lda.w'*csp_train.var_writing;
lda.projected_eating_train  = lda.w'*csp_train.var_eating;

lda.T = (mean(lda.projected_writing_train) + mean(lda.projected_eating_train) )/ 2;

%% classifying
lda.projection_train = lda.w'*csp_train.data;
lda.projection_test  = lda.w'*csp_test.data;

lda.classify_train = (lda.projection_train > lda.T)*1 + (lda.projection_train < lda.T)*2;
lda.classify_test  = (lda.projection_test > lda.T)*1 + (lda.projection_test < lda.T)*2;

accuracy_train_lda_custom = sum(csp_train.labels==lda.classify_train)/size(lda.classify_train,2)*100.0
accuracy_test_lda_custom  = sum(csp_test.labels==lda.classify_test)/size(lda.classify_test,2)*100.0

%% LDA matlab
lda_trainer = fitcdiscr(csp_train.data', csp_train.labels');
lda_train_matlab = predict(lda_trainer, csp_train.data');
lda_test_matlab  = predict(lda_trainer, csp_test.data');

accuracy_train_lda_matlab = sum(csp_train.labels'==lda_train_matlab)/size(lda_train_matlab,1)*100.0
accuracy_test_lda_matlab  = sum(csp_test.labels'==lda_test_matlab)/size(lda_test_matlab,1)*100.0

%% SVM
% order  = 2
svm_trainer_2 = fitcsvm(csp_train.data',csp_train.labels', ...
	'KernelFunction', 'polynomial', 'PolynomialOrder', 2);

svm_train_matlab = predict(svm_trainer_2, csp_train.data');
svm_test_matlab  = predict(svm_trainer_2, csp_test.data');

accuracy_train_svm_order2 = sum(csp_train.labels'==svm_train_matlab)/size(svm_train_matlab,1)*100.0
accuracy_test_svm_order2  = sum(csp_test.labels'==svm_test_matlab)/size(svm_test_matlab,1)*100.0

% order = 6
svm_trainer_6 = fitcsvm(csp_train.data',csp_train.labels', ...
	'KernelFunction', 'polynomial', 'PolynomialOrder', 6);

svm_train_matlab = predict(svm_trainer_6, csp_train.data');
svm_test_matlab  = predict(svm_trainer_6, csp_test.data');

accuracy_train_svm_order6 = sum(csp_train.labels'==svm_train_matlab)/size(svm_train_matlab,1)*100.0
accuracy_test_svm_order6  = sum(csp_test.labels'==svm_test_matlab)/size(svm_test_matlab,1)*100.0

%% plotting
% topolot
load('chanloc.mat');
figure
set(gcf, 'Position', [5000, 5000, 1000, 2000])
for i = 1:1:6
	subplot(2,3,i)
	topoplot(csp_train.w(:,i),chanlocs,'style','both','electrodes','labelpoint');
	title(i)
end

% % plotting LDA projections
% figure
% plot(lda.projected_writing_train)
% hold on
% plot(lda.projected_eating_train)
% 
% % custom LDA test classification
% figure
% set(gcf, 'Position', [250, 500, 500, 400])
% plot(lda.classify_test, 'r')
% hold on
% plot(csp_test.labels, 'b')
% title('custom LDA test classification');
% 
% % custom LDA train classification
% figure
% set(gcf, 'Position', [250, 50, 500, 400])
% plot(lda.classify_train, 'r')
% hold on
% plot(csp_train.labels, 'b')
% title('custom LDA train classification');
% 
% % MATLAB LDA test classification
% figure
% set(gcf, 'Position', [750, 500, 500, 400])
% plot(lda_test_matlab)
% hold on
% plot(csp_test.labels)
% title('MATLAB LDA test classification');
% 
% % MATLAB LDA train classification
% figure
% set(gcf, 'Position', [750, 50, 500, 400])
% plot(lda_train_matlab)
% hold on
% plot(csp_train.labels)
% title('MATLAB LDA train classification');
% 
% figure
% plot(accel.time, accel.Accelerometer)
% hold on
% plot(writing_window(:), 0, 'b*')
% plot(eating_window(:), 0, 'bo')
% plot(accel.time, accel.labels)
% 
% figure
% for i = 1:1:8
% 	subplot(2,4,i)
% 	plot(eeg.time, eeg.data(i,:))
% 	hold on
% 	plot(eeg.time, eeg.data_filt(i,:))
% 	plot(eeg.time, eeg.labels*mean(eeg.data_filt(i,:)))
% end
% 
% figure
% for i = 1:1:8
% 	subplot(2,4,i)
% 	plot([1:1:27027], eeg.writing(i,:))
% end
% 
% figure
% for i = 1:1:8
% 	subplot(2,4,i)
% 	plot([1:1:27027], eeg.eating(i,:))
% end
