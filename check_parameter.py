import torch
import os

def count_model_parameters(model_path, verbose=True):
    """
    Count the number of parameters in a PyTorch model saved as .pth file
    
    Args:
        model_path (str): Path to the .pth model file
        verbose (bool): Whether to print detailed parameter info
    
    Returns:
        dict: Dictionary containing parameter counts
    """
    # Check if file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load the model state dict
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
            
    except Exception as e:
        raise RuntimeError(f"Error loading model: {e}")
    
    # Count parameters
    total_params = 0
    trainable_params = 0
    
    param_info = {}
    
    for name, param in state_dict.items():
        num_params = param.numel()
        total_params += num_params
        
        # Assume all loaded parameters are trainable
        trainable_params += num_params
        
        param_info[name] = {
            'shape': list(param.shape),
            'params': num_params
        }
        
        if verbose:
            print(f"{name}: {list(param.shape)} -> {num_params:,} parameters")
    
    results = {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'parameter_info': param_info
    }
    
    if verbose:
        print(f"\n{'='*50}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB (float32)")
    
    return results

def main():
    # Example usage
    model_path = input("Enter the path to your .pth model file: ")
    
    try:
        results = count_model_parameters(model_path, verbose=True)
        
        # Save results to file
        output_file = "parameter_count.txt"
        with open(output_file, 'w') as f:
            f.write(f"Model: {model_path}\n")
            f.write(f"Total parameters: {results['total_parameters']:,}\n")
            f.write(f"Trainable parameters: {results['trainable_parameters']:,}\n")
            f.write(f"Model size: {results['total_parameters'] * 4 / 1024 / 1024:.2f} MB\n")
        
        print(f"\nResults saved to {output_file}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()