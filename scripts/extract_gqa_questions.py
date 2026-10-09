import urllib.request
import struct
import io
import zipfile

url = 'https://downloads.cs.stanford.edu/nlp/data/gqa/questions1.2.zip'

# 1. Get file size
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0', 'Range': 'bytes=0-0'})
with urllib.request.urlopen(req) as resp:
    cr = resp.headers.get('Content-Range')
    total_size = int(cr.split('/')[-1])
print(f"Total zip size: {total_size} bytes ({total_size / (1024**3):.2f} GB)")

# 2. Read last 128KB to parse EOCD
tail_len = min(131072, total_size)
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0', 'Range': f'bytes={total_size - tail_len}-{total_size - 1}'})
with urllib.request.urlopen(req) as resp:
    tail_data = resp.read()

eocd_idx = tail_data.rfind(b'PK\x05\x06')
if eocd_idx == -1:
    # Check for zip64
    z64_eocd_idx = tail_data.rfind(b'PK\x06\x06')
    print('Zip64 EOCD:', z64_eocd_idx)
    cd_size, cd_offset = struct.unpack('<QQ', tail_data[z64_eocd_idx + 40:z64_eocd_idx + 56])
else:
    disk_num, cd_disk, disk_entries, total_entries, cd_size, cd_offset = struct.unpack('<HHHHII', tail_data[eocd_idx+4:eocd_idx+20])
    if cd_offset == 0xFFFFFFFF or cd_size == 0xFFFFFFFF:
        z64_idx = tail_data.rfind(b'PK\x06\x06')
        print("Zip64 EOCD at:", z64_idx)
        cd_size, cd_offset = struct.unpack('<QQ', tail_data[z64_idx + 40:z64_idx + 56])

print(f"Central Directory offset: {cd_offset}, size: {cd_size}")

# 3. Read central directory
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0', 'Range': f'bytes={cd_offset}-{cd_offset + cd_size - 1}'})
with urllib.request.urlopen(req) as resp:
    cd_data = resp.read()

entries = []
pos = 0
while pos < len(cd_data):
    if cd_data[pos:pos+4] != b'PK\x01\x02':
        break
    (sig, v_made, v_needed, flags, method, mtime, mdate, crc, comp_size, uncomp_size,
     fn_len, extra_len, comment_len, disk, int_attr, ext_attr, local_offset) = struct.unpack(
        '<IHHHHHIIIHHHHII', cd_data[pos:pos+46])
    name = cd_data[pos+46:pos+46+fn_len].decode('utf-8', errors='ignore')
    
    # Check for zip64 extra field if needed
    extra = cd_data[pos+46+fn_len:pos+46+fn_len+extra_len]
    if uncomp_size == 0xFFFFFFFF or comp_size == 0xFFFFFFFF or local_offset == 0xFFFFFFFF:
        epos = 0
        while epos < len(extra):
            ehdr, esize = struct.unpack('<HH', extra[epos:epos+4])
            if ehdr == 1: # zip64
                zpos = epos + 4
                if uncomp_size == 0xFFFFFFFF:
                    uncomp_size = struct.unpack('<Q', extra[zpos:zpos+8])[0]
                    zpos += 8
                if comp_size == 0xFFFFFFFF:
                    comp_size = struct.unpack('<Q', extra[zpos:zpos+8])[0]
                    zpos += 8
                if local_offset == 0xFFFFFFFF:
                    local_offset = struct.unpack('<Q', extra[zpos:zpos+8])[0]
                    zpos += 8
                break
            epos += 4 + esize

    entries.append({'name': name, 'comp_size': comp_size, 'uncomp_size': uncomp_size, 'offset': local_offset, 'method': method})
    pos += 46 + fn_len + extra_len + comment_len

for e in entries:
    print(f"  {e['name']}: {e['comp_size'] / (1024**2):.1f} MB (uncomp: {e['uncomp_size'] / (1024**2):.1f} MB)")
