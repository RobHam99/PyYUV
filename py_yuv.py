import numpy as np
import os

class YUVFile:
    """
    Class for YUV file i/o and interplay with PyTorch.
    """

    def __init__(self, file_path: str, width: int, height: int, pixel_format: str = "yuv420p", bit_depth: int = 10, resolution: str = 'UHD'):
        """
        Initializes the YUV file.

        Args:
            file_path (str): Path to the YUV file.
            width (int): Frame width.
            height (int): Frame height.
            pixel_format (str): Chroma subsampling format (default is "yuv420p"). Supported: "yuv420p", "yuv422p", "yuv444p".
            bit_depth (int): Bit depth of the YUV file (default is 10).
        """
        self.file_path = file_path
        self.width = width
        self.height = height
        self.pixel_format = pixel_format
        self.bit_depth = bit_depth
        self.resolution = resolution
        if self.bit_depth == 8:
            self.dtype = np.uint8
            self.bit_depth_multiplier = 1
        elif self.bit_depth == 10:
            self.dtype = np.uint16
            self.bit_depth_multiplier = 2
        else:
            raise ValueError("Bit depth must be either 8 or 10.")

        self.y_size, self.uv_width, self.uv_height, self.uv_size = self._calculate_yuv_dimensions()
        self.frame_size = self._calculate_frame_size()

    def _calculate_yuv_dimensions(self):
        """
        Calculates the YUV dimensions based on the pixel format.

        Returns:
            tuple: Tuple (Y size, UV width, UV height).
        """
        y_size = self.width * self.height
        if self.pixel_format == "yuv420p":
            uv_width = self.width // 2
            uv_height = self.height // 2
        elif self.pixel_format == "yuv422p":
            uv_width = self.width // 2
            uv_height = self.height
        elif self.pixel_format == "yuv444p":
            uv_width = self.width
            uv_height = self.height
        else:
            raise ValueError("Unsupported pixel format. Supported: 'yuv420p', 'yuv422p', 'yuv444p'.")

        return y_size, uv_width, uv_height, uv_width * uv_height

    def _calculate_frame_size(self):
        """
        Calculates the size of a single frame in bytes.

        Returns:
            int: Size of a single frame in bytes.
        """

        yuv_dims = self.y_size + 2 * (self.uv_width * self.uv_height)
        return yuv_dims * self.bit_depth_multiplier

    def _calculate_seq_len(self):
        """
        Calculates the total number of frames in the YUV file.

        Returns:
            int: Total number of frames in the YUV file.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")
        file_size = os.path.getsize(self.file_path)
        return file_size // self.frame_size 
 
    def _binary_to_numpy(self, yuv_frames, num_frames):
        """
        Converts a binary frame to NumPy arrays.

        Args:
            frame (bytes): Binary frame data.

        Returns:
            tuple: Tuple (Y, U, V) as NumPy arrays.
        """
        np_frames = np.empty((num_frames, self.height, self.width, 3), dtype=np.float32)
        frame_size_16_bit = self.frame_size // self.bit_depth_multiplier
        for i in range(num_frames):
            frame_start_index = i * frame_size_16_bit
            frame_end_index = frame_start_index + frame_size_16_bit
            yuv_frame = yuv_frames[frame_start_index:frame_end_index]
            # Mask out unnecessary bits for 10-bit
            if self.bit_depth == 10:
                yuv_frame = yuv_frame & 0x3FF

            y = yuv_frame[0:self.y_size].reshape((self.height, self.width))
    
            u = yuv_frame[self.y_size:self.y_size + self.uv_size].reshape((self.uv_height, self.uv_width))
            v = yuv_frame[self.y_size + self.uv_size:].reshape((self.uv_height, self.uv_width))
            # Up-sample U and V channels to match Y channel size
            if self.pixel_format == 'yuv420p':
                u = np.repeat(np.repeat(u, 2, axis=0), 2, axis=1)
                v = np.repeat(np.repeat(v, 2, axis=0), 2, axis=1)

            elif self.pixel_format == 'yuv422p':
                u = np.repeat(u, 2, axis=1)
                v = np.repeat(v, 2, axis=1)

            np_frames[i] = np.stack((y, u, v), axis=2)
        return np_frames

    def load_yuv_frames(self, start_frame_id=0, num_frames=1):
        """
        Reads a single frame from the YUV file.

        Args:
            start_frame_id (int): Index of the frame to read (0-based).

        Returns:
            tuple: Tuple (Y, U, V) as NumPy arrays.
        """
        seq_len = self._calculate_seq_len()
        if start_frame_id >= seq_len:
            raise ValueError(f"Frame index {start_frame_id} is out of bounds. Total number of frames: {seq_len}")

        frame_start_bit = start_frame_id * self.frame_size
        with open(self.file_path, "rb") as yuv_file:
            yuv_file.seek(frame_start_bit, 0)
            yuv_frames = np.frombuffer(yuv_file.read(self.frame_size * num_frames), dtype=self.dtype)
        
        return self._binary_to_numpy(yuv_frames, num_frames)

    def normalise(self, frames):
        """
        Normalises the pixel values of the given frames based on the bit depth.

        Parameters:
        frames (numpy.ndarray): The input frames to be normalised.

        Returns:
        numpy.ndarray: The normalised frames with pixel values scaled to the range [0, 1].
        """
        normalisation_factor = (2 ** self.bit_depth) - 1
        return frames / normalisation_factor

    def unnormalise(self, frames):
        """
        Unnormalises the pixel values of the given frames based on the bit depth.

        Parameters:
        frames (numpy.ndarray): The input frames to be unnormalised.

        Returns:
        numpy.ndarray: The unnormalised frames with pixel values scaled to the range [0, 2**bit_depth - 1].
        """
        normalisation_factor = (2 ** self.bit_depth) - 1
        return frames * normalisation_factor

    def yuv_to_rgb(self, frames, normalise=False):
        """
        Converts YUV components to an RGB image.

        Args:
            y (numpy.ndarray): Luminance (Y) component.
            u (numpy.ndarray): Chrominance (U) component.
            v (numpy.ndarray): Chrominance (V) component.

        Returns:
            numpy.ndarray: RGB image as a NumPy array.
        """
        if frames[0].ndim != 3 or frames[0].shape != (self.height, self.width, 3):
            raise ValueError(f"Frame shape must be ({self.height}, {self.width}, 3). Found: {frames[0].shape}")
        
        frames = self.normalise(frames)
        y, u, v = frames[..., 0], frames[..., 1], frames[..., 2]
        
        # Conversion parameters
        if self.resolution == 'UHD':
            KR, KG, KB = 0.2627, 0.6780, 0.0593
            CU, CV = 1.8814, 1.4746
        elif self.resolution == 'HD':
            KR, KG, KB = 0.2126, 0.7152, 0.0722
            CU, CV = 1.7720, 1.5748 
        elif self.resolution == 'SD':
            KR, KG, KB = 0.299, 0.587, 0.114
            CU, CV = 1.772, 1.402
        else:
            raise ValueError("Unsupported resolution. Supported: 'UHD', 'HD', 'SD'.")

        u = u - 0.5
        v = v - 0.5
        # Conversion formulae
        r = y + CV * v
        b = y + CU * u
        g = (y - KR * r - KB * b) / KG

        # Clip pixel values to [0, 1]
        rgb = np.stack([r, g, b], axis=-1)
        rgb = np.clip(rgb, 0, 1)
        if not normalise:
            rgb = self.unnormalise(rgb)

        return rgb

    def rgb_to_yuv(self, frames, normalise=False):
        """
        Converts RGB image to YUV components.

        Args:
            frames (numpy.ndarray): RGB image to be converted to YUV.
            normalise (bool): Whether to normalise the frames or not (default is False).

        Returns:
            numpy.ndarray: YUV image as a NumPy array.
        """
        if frames[0].ndim != 3 or frames[0].shape != (self.height, self.width, 3):
            raise ValueError(f"Frame shape must be ({self.height}, {self.width}, 3). Found: {frames[0].shape}")
        
        frames = self.normalise(frames)
        r, g, b = frames[..., 0], frames[..., 1], frames[..., 2]

        # Conversion parameters
        if self.resolution == 'UHD':
            KR, KG, KB = 0.2627, 0.6780, 0.0593
            CU, CV = 1.8814, 1.4746
        elif self.resolution == 'HD':
            KR, KG, KB = 0.2126, 0.7152, 0.0722
            CU, CV = 1.7720, 1.5748 
        elif self.resolution == 'SD':
            KR, KG, KB = 0.299, 0.587, 0.114
            CU, CV = 1.772, 1.402
        else:
            raise ValueError("Unsupported resolution. Supported: 'UHD', 'HD', 'SD'.")

        # RGB to YUV conversion
        y = KR * r + KG * g + KB * b
        u = (b - y) / CU + 0.5 
        v = (r - y) / CV + 0.5

        # Stack Y, U, V channels together
        yuv = np.stack([y, u, v], axis=-1)
        yuv = np.clip(yuv, 0, 1)
        if not normalise:
            yuv = self.unnormalise(yuv)

        return yuv

    def save_yuv_frames(self, frames, output_path):
        """
        Saves the given frames to a YUV file.

        Args:
            frames (numpy.ndarray): Frames to be saved.
            output_path (str): Path to save the YUV file.
        """
        if frames.ndim != 4 or frames.shape[1:] != (self.height, self.width, 3):
                raise ValueError(f"Frame array shape must be (num_frames, {self.height}, {self.width}, 3). Found: {frames.shape}")
        yuv_data = []
        for frame in frames:
            frame = np.round(frame).astype(np.uint16)
            y = frame[:, :, 0]
            u = frame[:, :, 1]
            v = frame[:, :, 2]

            if self.pixel_format == 'yuv420p':
                u_downsampled = u[::2, ::2]
                v_downsampled = v[::2, ::2]
            elif self.pixel_format == 'yuv422p' or self.pixel_format == 'yuv444p':
                u_downsampled = u
                v_downsampled = v
            else:
                raise ValueError("Unsupported pixel format. Supported: 'yuv420p', 'yuv422p', 'yuv444p'.")

            yuv_frame = np.concatenate([y.flatten(), u_downsampled.flatten(), v_downsampled.flatten()])
            yuv_data.append(yuv_frame)

        yuv_data = np.concatenate(yuv_data)
        with open(output_path, "wb") as yuv_file:
            yuv_file.write(yuv_data.tobytes())
