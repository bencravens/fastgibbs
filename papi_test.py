from pypapi import papi_high

# Starts counters
papi_high.flops()  # -> Flops(0, 0, 0, 0)

# Read values
result = papi_high.flops()  # -> Flops(rtime, ptime, flpops, mflops)
print(result.mflops)

# Stop counters
papi_high.stop_counters()   # -> []
