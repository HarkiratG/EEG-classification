function labeled_data = label_data(time, writing_window, eating_window)

writing_index = 1;
eating_index = 1;
for i = 1:1:size(time,2)
	if writing_index <= size(writing_window,1) && time(i) >= writing_window(writing_index,1)
		labeled_data(i) = 1;
		if time(i) >= writing_window(writing_index,2)
			writing_index = writing_index+1;
		end
	elseif eating_index <= size(eating_window,1) && time(i) >= eating_window(eating_index,1)
		labeled_data(i) = 2;
		if time(i) >= eating_window(eating_index,2)
			eating_index = eating_index+1;
		end
	else
		labeled_data(i) = 0;
	end
end