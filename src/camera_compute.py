"""CPU/CUDA projection batches and bounded translation scoring."""

import numpy as np
from scipy.fft import next_fast_len
from scipy.ndimage import affine_transform


# These are the discovery failures for which CPU execution is an expected
# fallback.  CuPy exposes the same CUDA Runtime enum values on its runtime
# module; the numeric defaults keep the classification working with minimal
# test doubles and older CuPy builds.  See the CUDA Runtime API error enum:
# https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html
_CUDA_DISCOVERY_UNAVAILABLE_STATUSES = {
    "cudaErrorInsufficientDriver": 35,
    "cudaErrorNoDevice": 100,
}


def _is_known_cuda_discovery_failure(runtime, error):
    status = getattr(error, "status", None)
    return any(status == getattr(runtime, name, default)
               for name, default in _CUDA_DISCOVERY_UNAVAILABLE_STATUSES.items())


class GPUOutOfMemory(RuntimeError):
    """The smallest projection batch cannot fit on the selected GPU."""


class CameraCompute:
    def __init__(self, volume, images, weights, *, device, batch_size, warning_callback):
        self.xp = np
        self.transform = affine_transform
        self.device = "cpu"
        self.oom_errors = ()
        self.batch_size = batch_size
        self.warning_callback = warning_callback
        if device != "cpu":
            try:
                import cupy as cp
                from cupyx.scipy.ndimage import affine_transform as cuda_transform
            except (ImportError, OSError) as error:
                if device == "cuda":
                    raise RuntimeError("CUDA search requires a usable CuPy installation") from error
                warning_callback("GPU unavailable (CuPy not installed); using CPU.")
            else:
                reason = None
                discovery_error = None
                try:
                    available = cp.cuda.runtime.getDeviceCount() > 0
                except cp.cuda.runtime.CUDARuntimeError as error:
                    if not _is_known_cuda_discovery_failure(cp.cuda.runtime, error):
                        raise
                    available = False
                    discovery_error = error
                    reason = str(error)
                if not available:
                    if device == "cuda":
                        if reason is not None:
                            raise RuntimeError(f"No usable CUDA device: {reason}") from discovery_error
                        raise RuntimeError("No usable CUDA device")
                    if reason is not None:
                        warning_callback(f"GPU unavailable ({reason}); using CPU.")
                    else:
                        warning_callback("GPU unavailable (no usable CUDA device); using CPU.")
                else:
                    self.xp = cp
                    self.transform = cuda_transform
                    self.device = "cuda"
                    self.oom_errors = (cp.cuda.memory.OutOfMemoryError,)
        xp = self.xp
        try:
            self.volume = xp.asarray(volume, dtype=xp.float64)
            self.weights = xp.asarray(weights, dtype=xp.float64)
            self.weight_sum = self.weights.sum()
            self.size = volume.shape[0]
            self.fft_shape = (next_fast_len(2*self.size-1),) * 2
            self.weights_fft = xp.fft.rfft2(self.weights, s=self.fft_shape)
            self.targets = {}
            for key, image in images.items():
                target = xp.asarray(image, dtype=xp.float64)
                total = xp.sum(self.weights * target)
                variance = xp.sum(self.weights * target**2) - total**2/self.weight_sum
                self.targets[key] = (xp.fft.rfft2(self.weights * target, s=self.fft_shape),
                                     total, variance)
        except self.oom_errors as error:
            raise GPUOutOfMemory("GPU cannot hold the input volume and class images") from error

    def batches(self, matrices, keys, bound):
        """Yield rotations and per-class scores, retrying only allocation errors."""
        start = 0
        while start < len(matrices):
            stop = min(start + self.batch_size, len(matrices))
            try:
                scores = self._evaluate(matrices[start:stop], keys, bound)
            except self.oom_errors as error:
                if self.batch_size == 1:
                    raise GPUOutOfMemory("GPU out of memory at batch size 1") from error
                self.batch_size = max(1, self.batch_size // 2)
                self.warning_callback(f"GPU out of memory; retrying with batch size {self.batch_size}.")
                self.xp.get_default_memory_pool().free_all_blocks()
                continue
            yield matrices[start:stop], scores
            start = stop

    def _evaluate(self, matrices, keys, bound):
        xp = self.xp
        center = (self.size - 1) / 2
        projections = []
        for matrix in matrices:
            inverse = xp.asarray(matrix.T[::-1, ::-1])
            offset = center - inverse.sum(axis=1)*center
            rotated = self.transform(self.volume, matrix=inverse, offset=offset,
                                     order=1, mode="constant", cval=0., prefilter=False)
            projections.append(rotated.sum(axis=0))
        projections = xp.stack(projections)
        reverse = projections[:, ::-1, ::-1]
        reference_fft = xp.fft.rfft2(reverse, s=self.fft_shape)
        sums = xp.fft.irfft2(reference_fft*self.weights_fft, s=self.fft_shape)
        squares = xp.fft.irfft2(xp.fft.rfft2(reverse**2, s=self.fft_shape)*self.weights_fft,
                                s=self.fft_shape)
        variance = xp.maximum(0, squares - sums**2/self.weight_sum)
        mid = self.size-1
        result = {}
        for key in keys:
            target_fft, target_sum, target_var = self.targets[key]
            cross = xp.fft.irfft2(reference_fft*target_fft, s=self.fft_shape)
            denominator = xp.sqrt(xp.maximum(0, target_var*variance))
            score = (cross - target_sum*sums/self.weight_sum) / xp.maximum(denominator, 1e-30)
            score = xp.where(denominator > 1e-12, xp.clip(score, -1., 1.), -xp.inf)
            bounded = score[:, mid-bound:mid+bound+1, mid-bound:mid+bound+1].reshape(len(matrices), -1)
            indices = xp.argmax(bounded, axis=1)
            rows = xp.arange(len(matrices))
            values = xp.stack((bounded[rows, indices], indices % (2*bound+1)-bound,
                               indices // (2*bound+1)-bound), axis=1)
            result[key] = np.asarray(values) if xp is np else xp.asnumpy(values)
        return result
