api=tushare;

% get the current time and trace back to 2020-01-01 to get the index weight
% of the index code 000905.SH
time         = datetime(2021,01,01);
target_time  = time - years(5);
index_code   = '000905.SH';

% init index_weight
index_weight = [];

while time > target_time
    % pause due to not being able to access tushare more than 10 times per 
    % minutes
    %pause(6)
    if (time - years(0.5)) <= target_time
        % when the remaining days between 2016-01-01 and trace back time is
        % less than or equal to half a year, use 2016-01-01 as input of
        % api.get()
        index_weight_tmp = api.get('index_weight','index_code', ...
            index_code, 'start_date', yyyymmdd(target_time), ...
            'end_date',yyyymmdd(time));
        index_weight = vertcat(index_weight,index_weight_tmp);
        break
    else
        index_weight_tmp = api.get('index_weight','index_code', ...
            index_code, 'start_date', yyyymmdd(time - years(0.5)), ...
            'end_date',yyyymmdd(time));
        time = time - years(0.5);
        index_weight = vertcat(index_weight,index_weight_tmp);
    end
end

for i=["index_code" "con_code" "trade_date"]
    % convert index_code, con_code, trade_code to python readable dtype 
    index_weight.(i)=char(index_weight.(i));
end

index_weight = table2struct(index_weight,'ToScalar',true);
save(pwd + "\index_weight.mat",'index_weight');

