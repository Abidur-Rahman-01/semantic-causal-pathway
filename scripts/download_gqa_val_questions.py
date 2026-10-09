import io
import urllib.request
import zipfile
from pathlib import Path

class HttpSeekableStream(io.RawIOBase):
    def __init__(self, url):
        self.url = url
        self.pos = 0
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0', 'Range': 'bytes=0-0'})
        with urllib.request.urlopen(req) as resp:
            cr = resp.headers.get('Content-Range')
            self.length = int(cr.split('/')[-1])
        print(f"Remote file size: {self.length} bytes ({self.length / (1024**3):.2f} GB)")

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

out_dir = Path("data/raw/gqa")
out_dir.mkdir(parents=True, exist_ok=True)
dest = out_dir / "val_balanced_questions.json"

if dest.exists() and dest.stat().st_size > 1000000:
    print(f"{dest} already exists ({dest.stat().st_size / (1024**2):.1f} MB)")
else:
    print("Extracting val_balanced_questions.json from Stanford GQA zip (streaming 14.6 MB)...")
    url = "https://downloads.cs.stanford.edu/nlp/data/gqa/questions1.2.zip"
    stream = HttpSeekableStream(url)
    with zipfile.ZipFile(stream) as z:
        print("Reading val_balanced_questions.json from archive...")
        with z.open("val_balanced_questions.json") as src, dest.open("wb") as out:
            shutil_copy = True
            while chunk := src.read(512 * 1024):
                out.write(chunk)
    print(f"Extracted to {dest} ({dest.stat().st_size / (1024**2):.1f} MB)")
