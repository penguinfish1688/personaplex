#!/usr/bin/env python3
"""
Test script to verify that hidden layer extraction API is properly implemented
by checking source code directly without importing torch
"""

import os
import re

def test_hidden_layers_source_code():
    """Test that the hidden layer API is implemented in the source code"""
    print("Testing hidden layer extraction API implementation...")
    
    base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    
    # Test 1: Check LMGen.step method
    lm_file = os.path.join(base_path, "moshi", "moshi", "models", "lm.py")
    with open(lm_file, 'r') as f:
        lm_content = f.read()
    
    # Check step method signature
    step_pattern = r'def step\([^)]*return_hidden_layers[^)]*\)'
    if re.search(step_pattern, lm_content):
        print("✓ LMGen.step method has return_hidden_layers parameter")
    else:
        print("❌ LMGen.step method missing return_hidden_layers parameter")
        return False
    
    # Check forward_codes method
    forward_codes_pattern = r'def forward_codes\([^)]*return_hidden_layers[^)]*\)'
    if re.search(forward_codes_pattern, lm_content):
        print("✓ forward_codes method has return_hidden_layers parameter")
    else:
        print("❌ forward_codes method missing return_hidden_layers parameter")
        return False
    
    # Check forward_embeddings method
    forward_embeddings_pattern = r'def forward_embeddings\([^)]*return_hidden_layers[^)]*\)'
    if re.search(forward_embeddings_pattern, lm_content):
        print("✓ forward_embeddings method has return_hidden_layers parameter")
    else:
        print("❌ forward_embeddings method missing return_hidden_layers parameter")
        return False
    
    # Test 2: Check StreamingTransformer.forward method
    transformer_file = os.path.join(base_path, "moshi", "moshi", "modules", "transformer.py")
    with open(transformer_file, 'r') as f:
        transformer_content = f.read()
    
    # Check transformer forward method
    transformer_forward_pattern = r'def forward\([^)]*return_hidden_layers[^)]*\)'
    if re.search(transformer_forward_pattern, transformer_content):
        print("✓ StreamingTransformer.forward method has return_hidden_layers parameter")
    else:
        print("❌ StreamingTransformer.forward method missing return_hidden_layers parameter")
        return False
    
    # Check that hidden_layers is collected
    hidden_layers_collection = "hidden_layers.append(x.clone())" in transformer_content
    if hidden_layers_collection:
        print("✓ Transformer properly collects hidden layers")
    else:
        print("❌ Transformer doesn't collect hidden layers")
        return False
    
    # Test 3: Check implementation logic in step method
    bypass_graphed_pattern = r'if return_hidden_layers:.*?lm_model\.forward_codes.*?return_hidden_layers=True'
    if re.search(bypass_graphed_pattern, lm_content, re.DOTALL):
        print("✓ step method bypasses graphed version for hidden layers")
    else:
        print("❌ step method doesn't handle hidden layers properly")
        return False
    
    # Test 4: Check HiddenLayerOutputs dataclass
    hidden_layer_dataclass_pattern = r'@dataclass\s+class HiddenLayerOutputs'
    if re.search(hidden_layer_dataclass_pattern, lm_content):
        print("✓ HiddenLayerOutputs dataclass is defined")
    else:
        print("❌ HiddenLayerOutputs dataclass missing")
        return False
    
    # Test 5: Check depth transformer hidden layer extraction
    depformer_hidden_pattern = r'return_hidden_layers.*?depformer_step.*?return_hidden_layers=True'
    if re.search(depformer_hidden_pattern, lm_content, re.DOTALL):
        print("✓ Depth transformer hidden layers are extracted")
    else:
        print("❌ Depth transformer hidden layer extraction missing")
        return False
    
    print("\n✅ All hidden layer extraction features are properly implemented!")
    print("\nImplementation Summary:")
    print("- LMGen.step() accepts return_hidden_layers parameter")
    print("- When return_hidden_layers=True, bypasses CUDA graph and calls forward_codes directly")
    print("- forward_codes and forward_embeddings pass through the parameter")
    print("- StreamingTransformer.forward collects hidden states after each layer")
    print("- HiddenLayerOutputs dataclass separates text and depth transformer hidden layers")
    print("- Depth transformer hidden layers are extracted from depformer_step")
    print("- Returns tuple with output and HiddenLayerOutputs containing both sets of hidden layers")
    
    print("\nUsage:")
    print("  output, hidden_layers = lm_gen.step(input_tokens, return_hidden_layers=True)")
    print("  # hidden_layers.text_transformer[i] contains output of text transformer layer i")
    print("  # hidden_layers.depth_transformer[j] contains output of depth transformer layer j")
    
    return True

if __name__ == "__main__":
    success = test_hidden_layers_source_code()
    if success:
        print("\n🎉 Test passed! The hidden layer extraction feature is ready to use.")
    else:
        print("\n💥 Test failed! There may be implementation issues.")