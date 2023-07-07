api=tushare;

stock_basic = api.stock_basic();

for i=["ts_code" "name" "list_status"]
    % convert ts_code, name, list_status to python readable dtype 
    stock_basic.(i)=char(stock_basic.(i));
end

% convert datetime to python readable dtype 
stock_basic.list_date = yyyymmdd(stock_basic.list_date);
stock_basic.delist_date = yyyymmdd(stock_basic.delist_date);

stock_basic = table2struct(stock_basic,'ToScalar',true);
save(pwd + "\stock_basic.mat", 'stock_basic')