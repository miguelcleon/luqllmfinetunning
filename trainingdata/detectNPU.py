import intel_npu_acceleration_library as npu
count = npu.get_device_count()
print(f"NPU devices visible: {count}")
for i in range(count):
    print(npu.get_device_properties(i))