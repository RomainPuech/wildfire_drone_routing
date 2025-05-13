import torch
import platform

def check_gpu_availability():
    """Checks for CUDA (NVIDIA) or MPS (Apple Silicon) GPUs."""
    
    cuda_available = torch.cuda.is_available()
    mps_available = False
    if hasattr(torch.backends, "mps"): # Check if MPS is available attribute exists
        mps_available = torch.backends.mps.is_available()

    print("--- GPU Check ---")
    if cuda_available:
        gpu_count = torch.cuda.device_count()
        print(f"✅ CUDA (NVIDIA GPU) is available!")
        print(f"   Number of GPUs: {gpu_count}")
        for i in range(gpu_count):
            print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"   Current CUDA device index: {torch.cuda.current_device()}")
    elif mps_available:
        print(f"✅ MPS (Apple Silicon GPU) is available!")
        # Note: MPS doesn't have named devices like CUDA
        print(f"   Device: Apple Silicon GPU (MPS)")
    else:
        print("❌ No compatible GPU found (neither CUDA nor MPS).")
        print("   PyTorch will use the CPU.")
    
    print(f"Platform: {platform.system()} {platform.release()} ({platform.processor()})")
    print("-----------------")

if __name__ == "__main__":
    check_gpu_availability()