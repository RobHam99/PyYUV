from py_yuv import YUVFile
import os

yuv_file = YUVFile(
    file_path='/mnt/d/datasets/bvi-sr/ORIG/orig-yuv/Boat_3840x2160_60fps_10bit_420.yuv', 
    width=3840, 
    height=2160, 
    pixel_format='yuv420p', 
    bit_depth=10,
    resolution='UHD'
    )

frames = yuv_file.load_yuv_frames(start_frame_id=0, num_frames=10)

print('LOADED')
frames = yuv_file.yuv_to_rgb(frames, normalise=False)
print('YUV => RGB')

frames = yuv_file.rgb_to_yuv(frames, normalise=False)
print('RGB => YUV')

yuv_file.save_yuv_frames(frames, '/mnt/e/datasets/bvi-aom/meeting_sequences/BOAT.yuv')

