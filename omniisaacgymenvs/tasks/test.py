
import warp as wp

wp.init()
wp.config.mode = "debug"
DEVICE = "cpu"
wp.set_device(DEVICE)

@wp.kernel
def inc_kernel(a: wp.array(dtype=float)):
    tid = wp.tid()
    a[tid] = a[tid] + 1.0

if __name__ == "__main__":
    def test_launches(n, device, do_sync=False):
        arr = wp.zeros(1, dtype=wp.float32, device=device)
        wp.synchronize()

        with wp.ScopedTimer("launches"):
            for _ in range(n):
                wp.launch(inc_kernel, dim=arr.size, inputs=[arr], device=device)

            if do_sync:
                wp.synchronize()
                print('Here you go buddy', arr)

    test_launches(1, DEVICE, True)
