"""Henry Hub volatility during solar forecast surprises (actual vs day-before STPPF)."""

from regime_vol_page import render  # puts pipeline/ on sys.path via ramp_page/app_lib

render("solar_error", "Solar surprise volatility", "solar surprise", "Solar forecast error",
       "hours near forecast + night",
       intro="Is Henry Hub front-month volatility different when ERCOT solar comes in well above or below its "
             "day-before forecast (latest STPPF published by 10:00 CT the day before)? An HMM on the hourly error "
             "(Solar forecast error page) labels each hour as above, near or below forecast; the label applies to "
             "the four 15-minute NG returns in that hour. 'More' and 'less solar than forecast' are each compared "
             "against hours near forecast plus night. ")
