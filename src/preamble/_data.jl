using JLD2
dat_fin = jldopen( string(pwd(), "/data/vix_data.jld2" ) )
data_fin = read(dat_fin, "log_vix")
data_fin = data_fin[(end-n+1):end]
dates_fin = read(dat_fin, "dates")[(end-n+1):end]
