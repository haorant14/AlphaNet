daily_dir       = dir('C:\Users\Administrator\Desktop\唐浩然-实习文件\AlphaNet_impl\stock_daily_data\daily\*.mat');
daily_basic_dir = dir('C:\Users\Administrator\Desktop\唐浩然-实习文件\AlphaNet_impl\stock_daily_data\daily_basic\*.mat');
daily__dir    = dir('C:\Users\Administrator\Desktop\唐浩然-实习文件\AlphaNet_impl\stock_daily_data\daily_\*.mat');

% convert each daily table to struct and convert ts_code and time to python
% readable dtype; save all daily struct to dir daily
% for i=1:numel(daily_dir)
%     clear daily
%     file_name = daily_dir(i).name;
%     path = daily_dir(i).folder;
%     daily = load(path+"\"+file_name).daily;
%     data = table2struct(daily,'ToScalar',true);
%     data.ts_code = char(data.ts_code);
%     data.time = yyyymmdd(daily.time);
%     daily = data;
%     save('.\daily'+"\"+file_name,'daily');
% end
% 
% % convert each daily_basic table to struct and convert ts_code and time to 
% % python readable dtype; save all daily_basic struct to dir daily_basic
% for i=1:numel(daily_basic_dir)
%     clear daily_basic
%     file_name = daily_basic_dir(i).name;
%     path = daily_basic_dir(i).folder;
%     daily_basic = load(path+"\"+file_name).daily_basic;
%     data = table2struct(daily_basic,'ToScalar',true);
%     data.ts_code = char(data.ts_code);
%     data.time = yyyymmdd(daily_basic.time);
%     daily_basic = data;
%     save('.\daily_basic'+"\"+file_name,'daily_basic');
% end

for i=1:numel(daily__dir)
    clear daily_
    file_name = daily__dir(i).name;
    path = daily__dir(i).folder;
    daily_ = load(path+"\"+file_name).daily_;
    data = table2struct(daily_,'ToScalar',true);
    data.ts_code = char(data.ts_code);
    data.time = yyyymmdd(daily_.time);
    daily_ = data;
    save('.\daily_'+"\"+file_name,'daily_');
end