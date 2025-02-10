import os
#yuv_file = "/mnt/e/datasets/bvi-aom/meeting_sequences/mitchx8.yuv"
yuv_file = "/mnt/e/datasets/bvi-aom/ref-yuv/AAmericanFootballS3Harmonics_3840x2176_60fps_10bit_420.yuv"
#yuv_file = "/mnt/e/datasets/bvi-aom/patches_60f_test/AmericanFootballS3Harmonics/x2/bicubic_x2/patch_0/dis.yuv"
# For YUV420p with 10-bit depth
frame_size = (3840 * 2176) * 2 + (3840 * 2176 // 4) * 2 * 2  # Y, U, V channels (each 2 bytes per pixel)

total_frames = os.path.getsize(yuv_file) // frame_size
print(f"Total frames: {total_frames}")
