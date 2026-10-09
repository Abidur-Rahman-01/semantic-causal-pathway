import urllib.request
import re

html = urllib.request.urlopen('https://cs.stanford.edu/people/dorarad/gqa/download.html', timeout=10).read().decode('utf-8', errors='ignore')
for link in re.findall(r'https?://[^\s"\'<>]+\.zip', html):
    print(link)
