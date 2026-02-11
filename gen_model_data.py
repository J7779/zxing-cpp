#!/usr/bin/env python3
"""Generate C++ byte array from ONNX model file."""
import os

model_path = r'C:\Users\Rex\Documents\GitHub\zxing-cpp\nanodet_barcode.onnx'
output_path = r'C:\Users\Rex\Documents\GitHub\zxing-cpp\core\src\onnx\NanoDetModelData.cpp'

os.makedirs(os.path.dirname(output_path), exist_ok=True)

with open(model_path, 'rb') as f:
    data = f.read()

print(f'Model size: {len(data)} bytes')

lines = []
lines.append('// Auto-generated from nanodet_barcode.onnx - DO NOT EDIT')
lines.append('// This file embeds the NanoDet barcode detection ONNX model as a C++ byte array.')
lines.append('')
lines.append('#include "NanoDetModelData.h"')
lines.append('')
lines.append(f'const unsigned int NANODET_MODEL_SIZE = {len(data)};')
lines.append('')
lines.append(f'const unsigned char NANODET_MODEL_DATA[{len(data)}] = {{')

# Write in rows of 20 bytes for speed
for i in range(0, len(data), 20):
    chunk = data[i:i+20]
    hex_vals = ','.join(f'0x{b:02x}' for b in chunk)
    if i + 20 < len(data):
        lines.append(hex_vals + ',')
    else:
        lines.append(hex_vals)

lines.append('};')

with open(output_path, 'w') as out:
    out.write('\n'.join(lines))

print(f'Done. Output: {os.path.getsize(output_path)} bytes')
