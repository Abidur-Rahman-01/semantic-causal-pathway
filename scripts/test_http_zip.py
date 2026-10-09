import io
import urllib.request
import zipfile

class HttpSeekableStream(io.RawIOBase):
    def __init__(self, url):
        self.url = url
        self.pos = 0
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0', 'Range': 'bytes=0-0'})
        with urllib.request.urlopen(req) as resp:
            cr = resp.headers.get('Content-Range')
            self.length = int(cr.split('/')[-1])
        print(f"Remote file size: {self.length} bytes")

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos
    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET: self.pos = offset
        elif whence == io.SEEK_CUR: self.pos += offset
        elif whence == io.SEEK_END: self.pos = self.length + offset
        return self.pos

    def readinto(self, b):
        if self.pos >= self.length: return 0
        end = min(self.pos + len(b) - 1, self.length - 1)
        req = urllib.request.Request(self.url, headers={'User-Agent': 'Mozilla/5.0', 'Range': f'bytes={self.pos}-{end}'})
        with urllib.request.urlopen(req) as resp:
            data = resp.read()
        b[:len(data)] = data
        self.pos += len(data)
        return len(data)

stream = HttpSeekableStream('https://downloads.cs.stanford.edu/nlp/data/gqa/questions1.2.zip')
with zipfile.ZipFile(stream) as z:
    for name in z.namelist():
        info = z.getinfo(name)
        print(f"File: {name} (size: {info.file_size / (1024**2):.1f} MB, compressed: {info.compress_size / (1024**2):.1f} MB)")
