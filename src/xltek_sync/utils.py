from datetime import timedelta


# calculate sync time shift from video time since there maybe a precise 4 or 5 hour difference (timezone issue)
def timezone_shift(vid_time, sys_time):
    shift = vid_time - sys_time
    return timedelta(hours=(shift.total_seconds() // 3600))


# get sample from video time
def time_to_sample(vid_time, snc_time, snc_sample, frame_rate):
    return round(snc_sample + (vid_time-snc_time).total_seconds()*frame_rate)
