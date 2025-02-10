import numpy as np
#import numba
def rgb_to_yuv420p10bit(rgb_frames, output_file_path):
    num_frames, height, width, _ = rgb_frames.shape
    
    # List to store the YUV data per frame
    yuv_data = []

    for frame in rgb_frames:
        # Convert RGB to YUV using the common RGB -> YUV conversion matrix
        yuv_frame = np.zeros((height, width, 3), dtype=np.float32)
        
        # Apply the RGB to YUV conversion matrix (for RGB in [0, 1]) for BT.2020
        yuv_frame[:, :, 0] = 0.2627 * frame[:, :, 0] + 0.6780 * frame[:, :, 1] + 0.0593 * frame[:, :, 2]  # Y
        yuv_frame[:, :, 1] = -0.1396 * frame[:, :, 0] - 0.3604 * frame[:, :, 1] + 0.5000 * frame[:, :, 2]  # U
        yuv_frame[:, :, 2] = 0.5000 * frame[:, :, 0] - 0.4600 * frame[:, :, 1] - 0.0400 * frame[:, :, 2]  # V

        yuv_frame = np.round(yuv_frame * 1023).astype(np.uint16)

        # Extract Y, U, V components
        y = yuv_frame[:, :, 0]
        u = yuv_frame[:, :, 1]
        v = yuv_frame[:, :, 2]

        # Step 2: Downsample U and V channels for YUV420 format
        u_downsampled = u[::2, ::2]
        v_downsampled = v[::2, ::2]

        # Flatten channels and append to YUV data for writing
        yuv_frame = np.concatenate([y.flatten(), u_downsampled.flatten(), v_downsampled.flatten()])
        yuv_data.append(yuv_frame)

    # Write YUV data to file as 10-bit values (2 bytes per sample)
    yuv_data = np.concatenate(yuv_data)
    with open(output_file_path, 'wb') as f:
        f.write(yuv_data.tobytes())


def yuv2rgb(image, bit_depth, normalize=True):
    """Convert image from YUV to RGB color space."""

    N = ((2**bit_depth)-1)

    Y = np.float32(image[:,:,0])
    U = np.float32(image[:,:,1])
    V = np.float32(image[:,:,2])

    # Normalize Y, U, V if required
    if normalize:
        Y = Y / N
        U = U / N
        V = V / N
    else:
        Y = np.clip(Y, 0, N)
        U = np.clip(U, 0, N)
        V = np.clip(V, 0, N)

    fy = Y
    fu = U - 0.5  # Shift U to range [-0.5, 0.5]
    fv = V - 0.5  # Shift V to range [-0.5, 0.5]

    # Parameters for RGB conversion
    KR = 0.2627
    KG = 0.6780
    KB = 0.0593 

    R = fy + 1.4746 * fv
    G = fy - 0.3437 * fu - 0.7111 * fv
    B = fy + 1.8814 * fu

    # Clip RGB values to [0, 1]
    R = np.clip(R, 0, 1)
    G = np.clip(G, 0, 1)
    B = np.clip(B, 0, 1)

    # Stack the RGB channels together
    rgb_image = np.array([R, G, B])

    # Transpose to get correct image shape (height, width, 3)
    rgb_image = np.swapaxes(rgb_image, 0, 2)
    rgb_image = np.swapaxes(rgb_image, 0, 1)

    # If not normalizing, scale back to 10-bit range
    if not normalize:
        rgb_image = rgb_image * N

    return rgb_image


def load_yuv_frame(video_file, idx, width, height, bit_depth=10, pixel_format='yuv420p', use_numba=False):
    """
    Load a YUV frame from a video file.

    Args:
        video_file (file object): The video file to read from.
        idx (int): The index of the frame to load.
        width (int): The width of the frame.
        height (int): The height of the frame.
        bit_depth (int): Bit depth of the video (default: 10).
        pixel_format (str): The pixel format of the video (default: 'yuv420p').

    Returns:
        torch.Tensor: The loaded frame as a PyTorch tensor.
    """
    # Calculate frame size based on pixel format
    if bit_depth == 10:
        multiplier = 2
        _dtype = np.uint16
    elif bit_depth == 8:
        multiplier = 1
        _dtype = np.uint8
    else:
        raise ValueError(f"Unsupported bit depth: {bit_depth}")

    wh = width * height
    wh_2 = wh // 2
    wh_4 = wh // 4
    h_2, w_2 = height // 2, width // 2
    if pixel_format == 'yuv420p':
        frame_size = wh * 1.5  # Y + U + V (U and V downsampled)
    elif pixel_format == 'yuv422p':
        frame_size = wh * 2 # Y + U + V (U and V not downsampled)
    elif pixel_format == 'yuv444p':
        frame_size = wh * 3  # Y + U + V (all full resolution)
    else:
        raise ValueError(f"Unsupported pixel format: {pixel_format}")

    # Seek to the specified frame and read the YUV data
    video_file.seek(int(frame_size * idx * multiplier), 0)
    yuv_frame = np.frombuffer(video_file.read(int(frame_size * multiplier)), dtype=_dtype)
    if bit_depth == 10:
        yuv_frame = yuv_frame & 0x03FF  # Convert 16-bit data to 10-bit

    # Check if we read enough data
    if len(yuv_frame) < frame_size:
        raise ValueError(f"Not enough data read for frame index {idx}. Expected {frame_size} bytes, got {len(yuv_frame)}.")

    # Load Y, U, and V components
    y = yuv_frame[0:wh].reshape((height, width))

    if pixel_format == 'yuv420p':
        u = yuv_frame[wh:wh + wh_4].reshape((h_2, w_2))
        v = yuv_frame[wh + wh_4:].reshape((h_2, w_2))
        # Upsample U and V channels to match Y channel size
        u = np.repeat(np.repeat(u, 2, axis=0), 2, axis=1)
        v = np.repeat(np.repeat(v, 2, axis=0), 2, axis=1)
    elif pixel_format == 'yuv422p':
        u = yuv_frame[wh:wh + wh_2].reshape((height, w_2))
        v = yuv_frame[wh + wh_2:].reshape((height, w_2))
        # Upsample U and V channels to match Y channel size
        u = u.repeat(2, axis=1)
        v = v.repeat(2, axis=1)
    elif pixel_format == 'yuv444p':
        u = yuv_frame[wh:wh * 2].reshape((height, width))
        v = yuv_frame[wh * 2:].reshape((height, width))

    # Stack the Y, U, and V components
    frame = np.stack((y, u, v), axis=2) 
    # Convert YUV to RGB
    frame = yuv2rgb(frame, bit_depth=bit_depth, normalize=True, use_numba=use_numba)
    return frame


#@numba.jit(nopython=True)
def process_yuv_frames(yuv_data, num_frames, width, height, bit_depth, pixel_format, frame_size, multiplier):
    wh = width * height
    wh_2 = wh // 2
    wh_4 = wh // 4
    h_2, w_2 = height // 2, width // 2

    frames = np.empty((num_frames, height, width, 3), dtype=np.float32)
    for i in range(num_frames):
        # Extract the portion of yuv_data corresponding to the current frame
        frame_start = i * frame_size
        frame_end = (i + 1) * frame_size
        frame_data = yuv_data[frame_start:frame_end]
        if bit_depth == 10:
            frame_data = frame_data & 0x03FF  # Convert 16-bit data to 10-bit

        # Load Y, U, and V components
        y = frame_data[:wh].reshape((height, width))
        if pixel_format == 'yuv420p':
            u = frame_data[wh:wh + wh_4].reshape((h_2, w_2))
            v = frame_data[wh + wh_4:].reshape((h_2, w_2))
            # Upsample U and V channels to match Y channel size
            u = np.repeat(np.repeat(u, 2, axis=0), 2, axis=1) 
            v = np.repeat(np.repeat(v, 2, axis=0), 2, axis=1)
        elif pixel_format == 'yuv422p':
            u = frame_data[wh:wh + wh_2].reshape((height, w_2))
            v = frame_data[wh + wh_2:].reshape((height, w_2))
            # Upsample U and V channels to match Y channel size
            u = u.repeat(2, axis=1)
            v = v.repeat(2, axis=1)
        elif pixel_format == 'yuv444p':
            u = frame_data[wh:wh * 2].reshape((height, width))
            v = frame_data[wh * 2:].reshape((height, width))

        # Stack the Y, U, and V components
        frame = np.stack((y, u, v), axis=2) 
        # Convert YUV to RGB
        frame = yuv2rgb(frame, bit_depth=bit_depth, normalize=True)
        frames[i] = frame
    return frames


def load_yuv_frames(video_file_path, start_idx, num_frames, width, height, bit_depth=10, pixel_format='yuv420p'):
    """
    Load a specified number of YUV frames from a video file.

    Args:
        video_file (file object): The video file to read from.
        start_idx (int): The starting index of the frame to load.
        num_frames (int): The number of frames to load.
        width (int): The width of the frame.
        height (int): The height of the frame.
        bit_depth (int): Bit depth of the video (default: 10).
        pixel_format (str): The pixel format of the video (default: 'yuv420p').

    Returns:
        list: A list of loaded frames as numpy arrays.
    """
    # Calculate frame size based on pixel format
    if bit_depth == 10:
        multiplier = 2
        _dtype = np.uint16
    elif bit_depth == 8:
        multiplier = 1
        _dtype = np.uint8
    else:
        raise ValueError(f"Unsupported bit depth: {bit_depth}")

    wh = width * height
    if pixel_format == 'yuv420p':
        frame_size = int(wh * 1.5)  # Y + U + V (U and V downsampled)
    elif pixel_format == 'yuv422p':
        frame_size = int(wh * 2)  # Y + U + V (U and V not downsampled)
    elif pixel_format == 'yuv444p':
        frame_size = int(wh * 3)  # Y + U + V (all full resolution)
    else:
        raise ValueError(f"Unsupported pixel format: {pixel_format}")

    # Read the data for all frames at once
    total_size = int(frame_size * num_frames * multiplier)

    with open(video_file_path, 'rb') as video_file:
        # Seek to the starting frame
        video_file.seek(int(frame_size * start_idx * multiplier), 0)
        yuv_data = np.frombuffer(video_file.read(total_size), dtype=_dtype)

    # Process each frame using the optimized combined function
    frames = process_yuv_frames(yuv_data, num_frames, width, height, bit_depth, pixel_format, frame_size, multiplier)
    
    return frames

def write_yuv_array(yuv_array, output_file_path):
    num_frames, height, width, _ = yuv_array.shape

    # List to store the YUV data per frame
    yuv_data = []
    for frame in yuv_array:
        frame = np.round(frame).astype(np.uint16)

        # Extract Y, U, V components
        y = frame[:, :, 0]
        u = frame[:, :, 1]
        v = frame[:, :, 2]


        # Step 2: Downsample U and V for YUV420 (subsampling by 2x)
        u_downsampled = u[::2, ::2]
        v_downsampled = v[::2, ::2]

        # Step 4: Flatten and concatenate Y, U, V
        yuv_frame = np.concatenate([y.flatten(), u_downsampled.flatten(), v_downsampled.flatten()])
        yuv_data.append(yuv_frame)

    # Write YUV data to file as 10-bit values (2 bytes per sample)
    yuv_data = np.concatenate(yuv_data)
    with open(output_file_path, 'wb') as f:
        f.write(yuv_data.tobytes())

frames = load_yuv_frames('/mnt/e/datasets/bvi-aom/ref-yuv/AFireS21Mitch_3840x2176_24fps_10bit_420.yuv',
                            start_idx=0, num_frames=64, width=3840, height=2176, bit_depth=10, pixel_format='yuv420p')  

rgb_to_yuv420p10bit(frames, '/mnt/e/datasets/bvi-aom/meeting_sequences/pyyuv_test7uhd.yuv')