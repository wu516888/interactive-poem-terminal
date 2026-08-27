"""
Z-Order Morton Index for 5D spatial-temporal data.
Lightweight, pure Python with optional numpy acceleration.
"""

import numpy as np


class MortonIndex:
    """
    5D Z-Order Morton encoder/decoder.
    Maps (t, x, y, z, k) -> single integer key via bit-interleaving.
    """

    def __init__(self, bits_per_dim=12):
        """
        Args:
            bits_per_dim: bits allocated per dimension (default 12).
                          Total key width = bits_per_dim * 5.
        """
        self.bits_per_dim = bits_per_dim
        self.max_val = (1 << bits_per_dim) - 1

    def encode(self, t, x, y, z, k):
        """
        Encode five integers into a single Morton key.

        Args:
            t, x, y, z, k: integer coordinates

        Returns:
            int: interleaved Morton code
        """
        coords = [int(t), int(x), int(y), int(z), int(k)]
        normalized = [c & self.max_val for c in coords]

        result = 0
        for bit in range(self.bits_per_dim):
            for dim in range(5):
                bit_val = (normalized[dim] >> bit) & 1
                result |= (bit_val << (bit * 5 + dim))
        return result

    def decode(self, code):
        """
        Decode a Morton key back to five coordinates.

        Args:
            code: integer Morton key

        Returns:
            tuple: (t, x, y, z, k)
        """
        coords = [0] * 5
        for bit in range(self.bits_per_dim):
            for dim in range(5):
                bit_val = (code >> (bit * 5 + dim)) & 1
                coords[dim] |= (bit_val << bit)
        return tuple(coords)

    def range_query(self, center, radius):
        """
        Generate neighbor keys for range query.
        Simplified: returns keys for a 3x3x3x3x3 neighborhood.

        Args:
            center: tuple (t, x, y, z, k)
            radius: step size for neighbor generation

        Returns:
            list of Morton keys
        """
        t, x, y, z, k = center
        neighbors = []
        for dt in (-1, 0, 1):
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        for dk in (-1, 0, 1):
                            key = self.encode(
                                t + dt * radius,
                                x + dx * radius,
                                y + dy * radius,
                                z + dz * radius,
                                k + dk * radius,
                            )
                            neighbors.append(key)
        return neighbors

    def batch_encode(self, coords_array):
        """
        Vectorized encode for numpy arrays.

        Args:
            coords_array: ndarray of shape (N, 5)

        Returns:
            ndarray of shape (N,) with Morton keys
        """
        coords_array = np.asarray(coords_array, dtype=np.int64)
        t, x, y, z, k = coords_array[:, 0], coords_array[:, 1], coords_array[:, 2], coords_array[:, 3], coords_array[:, 4]
        t &= self.max_val
        x &= self.max_val
        y &= self.max_val
        z &= self.max_val
        k &= self.max_val

        result = np.zeros(len(coords_array), dtype=np.uint64)
        for bit in range(self.bits_per_dim):
            result |= ((t >> bit) & 1).astype(np.uint64) << (bit * 5 + 0)
            result |= ((x >> bit) & 1).astype(np.uint64) << (bit * 5 + 1)
            result |= ((y >> bit) & 1).astype(np.uint64) << (bit * 5 + 2)
            result |= ((z >> bit) & 1).astype(np.uint64) << (bit * 5 + 3)
            result |= ((k >> bit) & 1).astype(np.uint64) << (bit * 5 + 4)
        return result


if __name__ == "__main__":
    idx = MortonIndex(bits_per_dim=12)

    # Demo: encode / decode round-trip
    key = idx.encode(t=1692000000, x=1024, y=2048, z=512, k=3)
    print(f"Encoded key: {key}")

    decoded = idx.decode(key)
    print(f"Decoded:     {decoded}")

    # Range query demo
    neighbors = idx.range_query(center=(1024, 2048, 512, 256, 3), radius=100)
    print(f"Range query: {len(neighbors)} neighbor keys generated")

    # Batch encode demo
    batch = np.random.randint(0, 4096, size=(1000, 5))
    keys = idx.batch_encode(batch)
    print(f"Batch encode: {len(keys)} keys, unique = {len(np.unique(keys))}")
