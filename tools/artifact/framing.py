"""V3 entry/continuation headers and file payload alignment shared by reader and writer."""

import struct

MAGIC = b"NINFER\x00\x03"
PART_MAGIC = b"NINPRT\x00\x03"
HEADER = struct.Struct("<8sQ16s")
PAYLOAD_ALIGNMENT = 4096
