import torch

def main():
    print("Hello from gpt2-image-captioning!")
    
    # Check cuda
    print("CUDA available:", torch.cuda.is_available())


if __name__ == "__main__":
    main()
