import pandas as pd
from urllib.parse import urlparse
from collections import Counter
import math
url='my.execpc.com/~pjsports/SPORTSCASTER/1977-79%20SPORTSCASTER.htm'
url=url.strip("'\"")
parsed_url = urlparse(url)

url_for_parsing = url if parsed_url.scheme in {"http","https"} else "http://"+url
parsed_url=urlparse(url_for_parsing)
print(parsed_url)
print(int(parsed_url.path.lower().endswith(".exe")))
