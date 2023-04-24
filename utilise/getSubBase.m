
function sub_database = getSubBase(database,pre_rand)

    sub_database=database;
    
    sub_database.imnum = length(pre_rand);
    sub_database.label = database.label(pre_rand);
    sub_database.path = database.path(pre_rand);
    
    

    